"""Create the stage-5 draft from the immutable shared-controller release."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'fasterrcnn_online_shared_v1'
PACKAGE = ROOT/'fasterrcnn_class_adaptive_v1'
OUT = ROOT/'reports/fasterrcnn_class_adaptive_v1'

def write(path, text):
    path.write_text(text, encoding='utf-8', newline='\n')

def main():
    assert not PACKAGE.exists() and not OUT.exists(), 'Preserve existing work'
    sums = {}
    for line in (SOURCE/'SHA256SUMS').read_text().splitlines():
        sha, name = line.split('  ', 1)
        assert hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() == sha
        sums[name] = sha
    PACKAGE.mkdir(); OUT.mkdir()
    for name in sums:
        if name in ('LOCAL_VALIDATION.json', 'README.md', 'protocol.json', 'controller.py', 'tests/test_controller.py'): continue
        target = PACKAGE/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/name, target)
    shutil.copyfile(SOURCE/'controller.py', PACKAGE/'shared_controller.py')
    shutil.copyfile(SOURCE/'protocol.json', PACKAGE/'online_shared_protocol.json')
    cfg = json.loads((SOURCE/'protocol.json').read_text())
    cfg.update(version='fasterrcnn_class_adaptive_v1', augmentation='online_class_adaptive_local',
               purpose='Stage 5: exact overlap-group class-set conditioning versus the frozen online shared controller',
               recipe_status='Frozen stage-5 engineering choices after examining the stage-4 development run; class-adaptive outcomes unseen')
    cfg['comparator'].update(online_shared_run='fasterrcnn_online_shared_seed42_v1',
        online_shared_protocol_sha256=hashlib.sha256((SOURCE/'protocol.json').read_bytes()).hexdigest(),
        online_shared_controller_sha256=hashlib.sha256((SOURCE/'controller.py').read_bytes()).hexdigest(),
        online_shared_records_archive_sha256='b30df4e003be14cf8a53059341451c5101261d98d32e58cdc94866c793f0a231')
    cfg['augmentation_policy']['schedule'] = 'One unaugmented warmup epoch; before epochs 2,4,...,20 update each exact group class-set policy from its subset of the same paired panel'
    cfg['augmentation_policy']['family_sampling'] = 'Class-set-specific eligible families, normalized class-balanced loss score, temperature 0.25 and exploration 0.2; identical keyed uniforms to online shared'
    cfg['controller'].update(version='online_class_adaptive_local_v1', condition='exact_group_class_set',
        interpretation='Only feedback pooling and policy routing change from online shared. Exact class-set support may be sparse; no pooled/shared backoff, stale-policy carry, extra probes or threshold relaxation. One seed is descriptive, not efficacy confirmation.',
        support='Within each exact class set, every represented class needs 8 distinct reference-detected selected-group objects across 4 source clusters. Never borrow support across class sets. Missing/unsupported/no-admissible sets use identity.')
    write(PACKAGE/'protocol.json', json.dumps(cfg, indent=2, sort_keys=True)+'\n')
    # Preserve the full shared-controller regression suite, targeting its frozen implementation.
    tests = (SOURCE/'tests/test_controller.py').read_text().replace('import controller as c', 'import shared_controller as c')
    tests = tests.replace("b.PACKAGE/'protocol.json'", "b.PACKAGE/'online_shared_protocol.json'")
    write(PACKAGE/'tests/test_shared_controller.py', tests)
    for name in ('submit.sh','run_job.sh','job.pbs'):
        text = (PACKAGE/name).read_text().replace('fasterrcnn_online_shared', 'fasterrcnn_class_adaptive')
        text = text.replace('rdd_frcnn_onshr','rdd_frcnn_clsad')
        text = text.replace('Online shared GPU', 'Class-adaptive GPU')
        text = text.replace("print('Smoke active families:', record['controller_updates'][0]['policy']['families'])",
            "print('Smoke class-set policies:', record['controller_updates'][0]['policy']['class_sets'])")
        write(PACKAGE/name, text)
    text = (PACKAGE/'baseline.py').read_text()
    text = text.replace('"controller.py", "probe_detector.py"', '"controller.py", "shared_controller.py", "online_shared_protocol.json", "probe_detector.py"')
    text = text.replace('("shared_protocol", "shared_exposure", "local_protocol")', '("shared_protocol", "shared_exposure", "local_protocol", "online_shared_protocol")')
    text = text.replace('json.loads((PACKAGE / "local_protocol.json").read_text()))',
        'json.loads((PACKAGE / "local_protocol.json").read_text()),\n                       json.loads((PACKAGE / "online_shared_protocol.json").read_text()))')
    write(PACKAGE/'baseline.py', text)
    text = (PACKAGE/'probe_runner.py').read_text()
    old = "summary = {f: {'level': v['level'], 'p': round(v['probability'], 6)} for v in result['policy']['families'] for f in [v['family']]}"
    assert old in text
    text = text.replace(old, "summary = c.policy_summary(result['policy'])")
    text = text.replace("families={summary or 'identity'}", "class_sets={summary}")
    write(PACKAGE/'probe_runner.py', text)
    text = (PACKAGE/'tests/test_online_resume.py').read_text()
    text = text.replace("bad[0]['policy']['families'][0]['level'] = 1", "bad[0]['policy']['class_sets']['0']['families'][0]['level'] = 1")
    write(PACKAGE/'tests/test_online_resume.py', text)
    # The unchanged numerical core is separately checked against the shared implementation.
    source = (SOURCE/'controller.py').read_text()
    core = source[source.index('def aggregate('):source.index('\n\ndef decision(')]
    core = core.replace('def aggregate(panel, settings, epoch):', 'def _aggregate_set(panel, settings, epoch, represented):')
    core = core.replace("if not panel or len({p['image'] for p in panel}) != len(panel):", "if len({p['image'] for p in panel}) != len(panel):")
    core = core.replace("class_objects = {c: [] for c in settings['classes']}", "class_objects = {c: [] for c in represented}")
    core = core.replace("if any(c not in class_objects for c in p['classes']) or set(p['views']) != keys:", "if any(c not in settings['classes'] for c in p['classes']) or set(p['views']) != keys:")
    core = core.replace("for index in positions:\n            class_objects", "if tuple(sorted({p['classes'][i] for i in positions})) != tuple(represented):\n            raise ValueError('Feedback belongs to a different exact class set')\n        for index in positions:\n            class_objects")
    write(PACKAGE/'class_set_feedback.py', '"""Shared numerical measurement rules applied within one exact class set."""\nimport math\nfrom shared_controller import OPERATIONS, FAMILIES, variants, view_key, family, _mean, validate_observation, digest\n\n\n'+core+'\n')
    print('Created stage-5 draft:', PACKAGE)

if __name__ == '__main__': main()
