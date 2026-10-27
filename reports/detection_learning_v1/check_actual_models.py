"""Local CPU test of both real pretrained models on synthetic data; not CCAST efficacy."""
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

root = Path(__file__).resolve().parents[2]
config_dir = root/'tmp/detection_learning_v1_weights/ultralytics_config'
config_dir.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(config_dir)
sys.path.insert(0,str(root/'tmp/detection_learning_v1_weights/ultralytics_8_4_182'))
sys.path.insert(0,str(root/'detection_learning_v1'))
sys.path.insert(0,str(root/'detection_learning_v1/tests'))
import torch
import checkpoints
from integration import write_json
import learning
import metrics
from models import build_model, infer
from rdd_data import prepare_example
from test_learning import example

torch.set_num_threads(2)
protocol = json.loads((root/'detection_learning_v1/protocol.json').read_text())
examples = [example(),example(1,False)]
for item in examples:
    item.geometry = None
examples = [prepare_example(e,128) for e in examples]
results = {'scope':'Actual pretrained construction, one update, inference and checkpoint reload on synthetic CPU inputs',
           'python':platform.python_version(),
           'versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow','pycocotools','ultralytics-opencv-headless')},
           'raw_rdd_images_read':0,'models':{}}
for name in protocol['models']:
    torch.manual_seed(42)
    spec = protocol['pretrained_weights'][name]
    path = root/'tmp/detection_learning_v1_weights'/spec['filename']
    assert checkpoints.file_hash(path) == spec['sha256']
    model, head, effective = build_model(name,128,'cpu',protocol['evaluation'],path)
    optimizer = learning.optimizer_for(model,protocol)
    update = learning.train_step(model,head,name,examples,optimizer,'cpu')
    gt = metrics.ground_truth(examples)
    measurement,predictions = learning.evaluate_model(model,name,[examples],protocol['evaluation'],'cpu',gt)
    with tempfile.TemporaryDirectory() as d:
        checkpoint = Path(d)/'test.pt'
        saved = checkpoints.save(checkpoint,model,optimizer,name,1,{'local':'test'})
        del optimizer,head,model
        gc.collect()
        model,head,_ = build_model(name,128,'cpu',protocol['evaluation'],None)
        optimizer = learning.optimizer_for(model,protocol)
        restored = checkpoints.restore(checkpoint,saved,model,optimizer,name,{'local':'test'})
        _,after = learning.evaluate_model(model,name,[examples],protocol['evaluation'],'cpu',gt)
        restored.update(learning.compare_predictions(predictions,after))
        next_update = learning.train_step(model,head,name,examples,optimizer,'cpu')
        empty_update = learning.train_step(model,head,name,[examples[1],examples[1]],optimizer,'cpu')
    results['models'][name] = {'status':'passed','effective_configuration':effective,
                              'first_update':update['update'],'continued_update':next_update['update'],
                              'all_empty_update':empty_update['update'],
                              'inference_prediction_boxes':len(predictions),'restore':restored}
    print(name,'actual pretrained CPU update/inference/reload/continued update: PASS',flush=True)
    del optimizer,head,model
    gc.collect()
results['status']='passed'
write_json(Path(__file__).with_name('local_model_check.json'),results)
