"""Freeze the independently tested controller; refuse to overwrite a release."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PACKAGE = ROOT/'fasterrcnn_online_shared_v1'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save_new(path, value):
    data = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(data)


def read(name):
    return json.loads((OUT/name).read_text())


def main():
    archive_path = ROOT/(PACKAGE.name+'.tar.gz')
    for path in (archive_path, PACKAGE/'LOCAL_VALIDATION.json', PACKAGE/'SHA256SUMS', OUT/'release.json'):
        assert not path.exists(), f'Preserve existing release: {path}'
    tests, actual, parity, plan, shell, tested = [read(name) for name in (
        'unit_tests.json', 'actual_model_check.json', 'comparator_checks.json', 'panel_plan.json', 'shell_checks.json', 'tested_sources.json')]
    assert tests['tests'] == 63 and not any(tests[k] for k in ('failures', 'errors', 'skipped'))
    assert actual['status'] == parity['status'] == shell['status'] == 'passed'
    assert actual['initialization_matches_identity_exactly'] and actual['standard_predictions_unchanged_before_after_probe']
    assert actual['fresh_model_checkpoint_prediction_policy_replay'] == 'exact' and actual['complete_probe_views'] == 112
    assert actual['model_buffers_gradients_rng_unchanged'] and actual['duplicate_and_missed_gt_losses_present']
    assert plan['scheduled_updates'] == 10 and plan['probe_image_forwards'] == 71680
    assert plan['total_application_opportunities'] == 158151 and plan['all_opportunities_match_fixed_controls']
    for name, expected in {**tested, **shell['source_sha256']}.items():
        assert digest((PACKAGE/name).read_bytes()) == expected, f'Changed after verification: {name}'
    cfg = json.loads((PACKAGE/'protocol.json').read_text())
    assert digest((PACKAGE/'protocol.json').read_bytes()) == parity['protocol_sha256']
    for name in parity['unchanged_dependency_files']:
        assert (PACKAGE/name).read_bytes() == (ROOT/'fasterrcnn_local_v1'/name).read_bytes()
    paths = sorted(p for p in PACKAGE.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    assert not any(p.is_symlink() for p in paths)
    for path in paths:
        data = path.read_bytes()
        assert b'\r' not in data and not data.startswith(b'\xef\xbb\xbf'), path
        if path.suffix == '.py': ast.parse(data.decode(), feature_version=(3, 12))
    evidence = ['unit_tests.json', 'unit_tests.txt', 'actual_model_check.json', 'comparator_checks.json', 'panel_plan.json', 'shell_checks.json', 'tested_sources.json']
    validation = {'status': 'passed_locally_ccast_pending', 'python': platform.python_version(),
                  'focused_tests': tests, 'actual_pretrained_model_check': actual, 'comparator_checks': parity,
                  'shell_syntax': shell, 'python_3_12_grammar': 'passed',
                  'planned_panels': {k: v for k, v in plan.items() if k != 'epochs'},
                  'panel_support_metadata': [{k: v for k, v in entry.items() if k != 'panel'} for entry in plan['epochs']],
                  'limits': ['No CCAST execution, RDD raw pixels, CUDA or new scientific results measured locally.',
                             'Synthetic actual-model panel had no qualifying detection support; identity fallback was exercised. Nontrivial policies and resume were tested with known synthetic feedback.',
                             'Local library versions differ; PBS repeats tests and real-data GPU smoke before full training.',
                             'One shared policy, not class-conditioned; full 20-epoch detector budget is unchanged but probes add compute.',
                             'The comparison bundles adaptation with the extra strengths. Single-seed small effects and pre-augmentation divergence do not establish efficacy.'],
                  'source_sha256': {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths},
                  'local_evidence_sha256': {name: digest((OUT/name).read_bytes()) for name in evidence}}
    save_new(PACKAGE/'LOCAL_VALIDATION.json', validation)
    paths.append(PACKAGE/'LOCAL_VALIDATION.json'); paths.sort()
    sums = {p.relative_to(PACKAGE).as_posix(): digest(p.read_bytes()) for p in paths}
    save_new(PACKAGE/'SHA256SUMS', ''.join(f'{h}  {name}\n' for name, h in sums.items()).encode())
    paths.append(PACKAGE/'SHA256SUMS'); paths.sort()
    with tarfile.open(archive_path, 'x:gz') as archive:
        for path in paths:
            archive.add(path, arcname=f'{PACKAGE.name}/{path.relative_to(PACKAGE).as_posix()}', recursive=False)
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert len(members) == len(paths) == len({m.name for m in members})
        for member in members:
            name = PurePosixPath(member.name)
            assert member.isfile() and name.parts[0] == PACKAGE.name and '..' not in name.parts
            assert archive.extractfile(member).read() == ROOT.joinpath(*name.parts).read_bytes()
    release = {'status': 'prepared_locally_ccast_submission_pending', 'archive': archive_path.name,
               'sha256': digest(archive_path.read_bytes()), 'bytes': archive_path.stat().st_size,
               'archive_files': len(paths), 'checksum_entries': len(sums), 'unit_tests_passed': tests['tests'],
               'actual_model_synthetic_cpu_check': 'passed', 'comparator_checks': 'passed',
               'condition': 'online_shared_local', 'protocol_sha256': digest((PACKAGE/'protocol.json').read_bytes()),
               'training_epochs': 20, 'seed': 42, 'detector_optimizer_steps': 134320, 'training_images': 26863,
               'validation_images': 5760, 'test_data_read': False, 'controller_updates': 10,
               'additional_committed_probe_image_forwards': 71680, 'fixed_application_opportunities': 158151,
               'full_run_directory': '/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/runs/fasterrcnn_online_shared_seed42_v1',
               'pbs': {'queue': 'gpus', 'cpus': 4, 'memory_gb': 32, 'gpus': 1, 'gpu_type': 'a10', 'walltime_hours': 24},
               'ccast_run': 'pending', 'ccast_tests': 'pending', 'ccast_smoke': 'pending'}
    save_new(OUT/'release.json', release)
    print(json.dumps(release, indent=2))


if __name__ == '__main__':
    main()
