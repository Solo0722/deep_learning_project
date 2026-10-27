"""Synthetic CPU check of the actual YOLO path, including a local checkpoint reload."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile

root=Path(__file__).resolve().parents[2]
config=root/'tmp/detection_learning_v1_weights/ultralytics_config'
config.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(config)
sys.path.insert(0,str(root/'tmp/detection_learning_v1_weights/ultralytics_8_4_182'))
sys.path.insert(0,str(root/'detection_learning_v1/tests'))
sys.path.insert(0,str(root/'detection_diagnostics_v1'))
import torch
import diagnose
import checkpoints
from integration import write_json
import learning
import metrics
from models import build_model
from rdd_data import prepare_example
from test_learning import example

torch.set_num_threads(2)
torch.manual_seed(42)
protocol=json.loads((root/'detection_learning_v1/protocol.json').read_text())
examples=[example(0,True),example(1,False),example(2,True)]
for item in examples:
    item.geometry=None
examples=[prepare_example(item,128) for item in examples]
weight=root/'tmp/detection_learning_v1_weights/yolov8n.pt'
assert checkpoints.file_hash(weight)==protocol['pretrained_weights']['yolov8n']['sha256']
model,_,effective=build_model('yolov8n',128,'cpu',protocol['evaluation'],weight)
optimizer=learning.optimizer_for(model,protocol)
original,predictions=learning.evaluate_model(model,'yolov8n',[examples[:2],examples[2:]],protocol['evaluation'],'cpu',metrics.ground_truth(examples))
context={'local':'synthetic diagnostic integration check'}
with tempfile.TemporaryDirectory() as temporary:
    path=Path(temporary)/'checkpoint.pt'
    saved=checkpoints.save(path,model,optimizer,'yolov8n',100,context)
    del optimizer,model
    model,_,_=build_model('yolov8n',128,'cpu',protocol['evaluation'],None)
    diagnose.load_existing(path,saved,context,model)
    result,replayed=diagnose.diagnose_checkpoint(model,examples,protocol['evaluation'],'cpu',predictions,
                                                ['running_stats','batch_stats','running_stats_after'])
    assert checkpoints.file_hash(path)==saved['sha256']
    assert result['running_stats']['metrics']==original['metrics']
    assert all(v['state_and_rng_unchanged'] for v in result.values())
    assert all(p.grad is None for p in model.parameters())
    assert result['running_stats']['batchnorm_modules']>0
    record={'status':'passed','python':platform.python_version(),
            'versions':{k:version(k) for k in ('torch','torchvision','numpy','Pillow','pycocotools','ultralytics-opencv-headless')},
            'scope':'Actual pretrained YOLO architecture on three synthetic 128-pixel images; includes a mixed batch and final single positive image',
            'raw_rdd_images_read':0,'optimizer_steps':0,'actual_ccast_checkpoints_read':0,
            'checkpoint_hash_unchanged':True,'all_parameters_grad_none':True,
            'baseline_matches_released_inference':True,'modes':result}
    write_json(Path(__file__).with_name('local_model_check.json'),record)
    print('ACTUAL YOLO DIAGNOSTIC CPU CHECK: PASS; parameters, BN buffers, RNG and checkpoint file unchanged')
