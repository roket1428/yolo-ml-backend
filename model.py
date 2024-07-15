import os, shutil, json, time, subprocess, glob

from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_image_size, is_skipped

from ultralytics.models import YOLO

TRAINED_MODEL_ = os.path.join(os.getenv("CHECKPOINT_DIR", "./yolobk/data/checkpoints"), "yolov10n_trained_.pt")
TRAINED_MODEL = os.path.join(os.getenv("CHECKPOINT_DIR", "./yolobk/data/checkpoints"), "yolov10n_trained.pt")
INITIAL_MODEL = os.path.join(os.getenv("CHECKPOINT_DIR", "./yolobk/data/checkpoints"), "yolov10n.pt")
BEST_MODEL = "./yolobk/runs/detect/train/weights/best.pt"

DATASET_DIR = os.getenv("DATASET_DIR", "./yolobk/dataset")
DATASET_ROOT = os.path.join(DATASET_DIR, "dataroot")
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "images/train")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "labels/train")
VAL_IMAGES = os.path.join(DATASET_ROOT, "images/val")
VAL_LABELS = os.path.join(DATASET_ROOT, "labels/val")


class YOLOv10(LabelStudioMLBase):
    """ Active learning setup for the YOLOv10 Model
    """
    def __init__(self, project_id: str | None = None, label_config=None):
        super().__init__(project_id, label_config)
        self.dataset_state = self.load_state()
        self.waiting_annotations = self.dataset_state["waiting_annotations"]
        # current data target: "train" or "val"
        self.current_target = self.dataset_state["current_target"]
        # the amount of data for training in this waiting annotations set
        self.for_training = self.dataset_state["for_training"]
        # the amount of data for validation in this waiting annotations set
        self.for_validation = self.dataset_state["for_validation"]

        print("initialized")
        # model settings
        self.workers = 8
        self.batch_size = 10
        self.imgsz = 128
        self.epochs = 10
        self.lr = 0.01

    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "0.0.1")

        if os.path.isfile(TRAINED_MODEL):
            self.current_model = TRAINED_MODEL
        else:
            self.current_model = INITIAL_MODEL

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Extra params: {self.extra_params}''')
        # Label config: {self.label_config}
        # Parsed JSON Label config: {self.parsed_label_config}

        task = tasks[0]
        model = YOLO(self.current_model)

        image_path = self.get_local_path(task['data']['image'], task_id=task['id'])
        image_width, image_height = get_image_size(image_path)
        results = model(image_path)
        boxes = results[0].boxes
        output = []
        max_conf = {}
        max_conf["minus"] = max_conf["plus"] = 0
        for (x_min, y_min, x_max, y_max), conf, cls in zip((boxes.xyxy.numpy()), boxes.conf.numpy(), boxes.cls.numpy()):
            current_class = model.names[cls]
            if not current_class in ["minus", "plus"]:
                continue

            if conf > max_conf[current_class]:
                max_conf[current_class] = conf

            output.append({
                "from_name": "label",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "type": "rectanglelabels",
                "value": {
                    "rectanglelabels": [current_class],
                    "x": float(x_min / image_width * 100),
                    "y": float(y_min / image_height * 100),
                    "width": float((x_max - x_min) / image_width * 100),
                    "height": float((y_max - y_min) / image_height * 100)
                },
                "score": float(conf)
            })

        all_scores = []
        output_processed = []
        for pred in output:
            if pred["score"] < max_conf[pred["value"]["rectanglelabels"][0]]:
                continue
            else:
                output_processed.append(pred)
                all_scores.append(pred["score"])

        avg_score = sum(all_scores) / max(len(all_scores), 1)
        predictions = [{
            "result": output_processed,
            "score": avg_score,
            "model_version": self.get("model_version")
        }]

        print(predictions)
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        if (event == "ANNOTATION_CREATED" or event == "ANNOTATION_UPDATED" or event == "START_TRAINING"):
            if (event == "ANNOTATION_CREATED" or event == "ANNOTATION_UPDATED"):

                # check if the current annotation is skipped
                data_wrapper = {}
                data_wrapper["annotations"] = [data["annotation"]]
                if is_skipped(data_wrapper):
                    return

                # update state
                if self.for_training > 0:
                    current_images = TRAIN_IMAGES
                    current_labels = TRAIN_LABELS
                    self.current_target = "train"
                else:
                    current_images = VAL_IMAGES
                    current_labels = VAL_LABELS
                    self.current_target = "val"

                task = data["task"]
                image_path = self.get_local_path(task['data']['image'], task_id=task['id'])
                image_name = image_path.split("/")[-1]

                if (event == "ANNOTATION_UPDATED"):
                    print("updating the annotation")
                    # dont count updates as new annotation
                    self.waiting_annotations -= 1
                    if self.current_target == "train":
                        self.for_training += 1
                        # if updated annotation isn't in the current set
                        if not os.path.isfile(os.path.join(current_images, image_name)):
                            current_images = VAL_IMAGES
                            current_labels = VAL_LABELS
                    else:
                        self.for_validation += 1
                        if not os.path.isfile(os.path.join(current_images, image_name)):
                            current_images = TRAIN_IMAGES
                            current_labels = TRAIN_LABELS

                # create the dataset at the target
                self.create_dataset(data, image_path, image_name, current_images, current_labels)

                # update state
                if self.for_training > 0:
                    self.for_training -= 1
                else:
                    self.for_validation -= 1
                self.waiting_annotations += 1

                # save state
                self.dataset_state["waiting_annotations"] = self.waiting_annotations
                self.dataset_state["current_target"] = self.current_target
                self.dataset_state["for_training"] = self.for_training
                self.dataset_state["for_validation"] = self.for_validation
                self.save_state(self.dataset_state)

            if self.waiting_annotations == 80:
                # check for human errors before training
                self.fix_human_errors_on_labelling()
                if os.path.isfile(BEST_MODEL):
                    shutil.copyfile(BEST_MODEL, TRAINED_MODEL_)
                    shutil.copytree("./yolobk/runs/detect/train", "./yolobk/archive/runs/" + str(int(time.time())))
                    shutil.rmtree("./yolobk/runs/detect")
                print("starting the training")
                subprocess.Popen(["yolo",
                                  "mode=train",
                                  f"workers={self.workers}",
                                  f"batch={self.batch_size}",
                                  f"imgsz={self.imgsz}",
                                  f"epochs={self.epochs}",
                                  f"model={self.current_model}",
                                  f"data={os.path.join(DATASET_DIR, "data.yaml")}"])
                print("training started as an async sub-process")
                # update the model version
                model_version = self.get("model_version")
                self.set("model_version", ".".join(model_version.split(".")[:-1]) + "." + str(int(model_version.split(".")[-1]) + 1))

                # reset state
                self.current_target = "train"
                self.waiting_annotations = 0
                self.for_training = 60
                self.for_validation = 20
                self.dataset_state["waiting_annotations"] = self.waiting_annotations
                self.dataset_state["current_target"] = self.current_target
                self.dataset_state["for_training"] = self.for_training
                self.dataset_state["for_validation"] = self.for_validation
                self.save_state(self.dataset_state)
                print("training probably continues as a subprocess but the training part of the fit function ended")

            print(event)
            print(data)
            print(self.waiting_annotations)
            print(self.for_training)
            print(self.for_validation)
            print(self.current_target)

    def fix_human_errors_on_labelling(self):
        """
        Fix wrong labelling on the human side.
        Since all images have class names in it we can use that to check incorrectly assingned class names.
        """
        plus_labels = glob.glob(os.path.join(TRAIN_LABELS, "p*.txt")) + glob.glob(os.path.join(VAL_LABELS, "p*.txt"))
        minus_labels = glob.glob(os.path.join(TRAIN_LABELS, "m*.txt")) + glob.glob(os.path.join(VAL_LABELS, "m*.txt"))
        label_classes = {}
        label_classes["minus"] = 0
        label_classes["plus"] = 1
        for label_path in plus_labels:
            with open(label_path) as f:
                label_txt = f.read()
            label_class = int(label_txt[0])
            if label_class != label_classes["plus"]:
                label_txt = str(label_classes["plus"]) + label_txt[1:]
                with open(label_path, "w") as f:
                    f.write(label_txt)
        for label_path in minus_labels:
            with open(label_path) as f:
                label_txt = f.read()
            label_class = int(label_txt[0])
            if label_class != label_classes["minus"]:
                label_txt = str(label_classes["minus"]) + label_txt[1:]
                with open(label_path, "w") as f:
                    f.write(label_txt)

    def create_dataset(self, data, image_path, image_name, images_dir, labels_dir):
        annotation = data["annotation"]
        if os.path.isfile(os.path.join(labels_dir, image_name[:-4]+'.txt')):
            os.remove(os.path.join(labels_dir, image_name[:-4]+'.txt'))
        for bbox in annotation['result']:
            bb_width = (bbox['value']['width']) / 100
            bb_height = (bbox['value']['height']) / 100
            x = (bbox['value']['x'] / 100 ) + (bb_width/2)
            y = (bbox['value']['y'] / 100 ) + (bb_height/2)
            label = bbox['value']['rectanglelabels']
            if label[0] == 'minus':
                label_idx = 0
            else:
                label_idx = 1

            with open(os.path.join(labels_dir, image_name[:-4]+'.txt'), 'a') as f:
                f.write(f"{label_idx} {x} {y} {bb_width} {bb_height}\n")
        shutil.copyfile(image_path, os.path.join(images_dir, image_name))

    def load_state(self):
        with open(os.path.join(DATASET_DIR, "dataset_state.json"), "r") as f:
            out = json.loads(f.read())
        return out

    def save_state(self, state):
        with open(os.path.join(DATASET_DIR, "dataset_state.json"), "w") as f:
            f.write(json.dumps(state))