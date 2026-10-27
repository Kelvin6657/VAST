"""Re-evaluate a recorded VAST best checkpoint on its internal validation split.

The prepared split, protocol, manifest and checkpoint are local, private inputs.
This command does not create a new split or measure external generalization.
"""
import argparse
import hashlib
import importlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_POLICY = 'Same oracle segmentation view mask and eligible classification slots in ALL stages'
REQUIRED_SOURCES = (
    'progressive.py', 'core.py', 'runtime/step_1_unimatch_train.py',
    'runtime/model/Echocare.py', 'runtime/dataset/fetus.py',
    'runtime/dataset/transform.py', 'runtime/util/utils.py',
)
METRICS = ('Dice', 'NSD', 'Macro-F1', 'score')


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Recorded best.pth from a completed progressive run; manifest.json must be adjacent.')
    parser.add_argument('--project-root', type=Path, required=True,
                        help='Private workspace used by VAST/progressive.py prepare and recorded in the protocol.')
    parser.add_argument('--device', default='cuda:0', help='PyTorch device; default: cuda:0. Use cpu explicitly if needed.')
    parser.add_argument('--output', type=Path,
                        help='New output JSON path; defaults to evaluation.json beside the checkpoint.')
    return parser.parse_args(argv)


def _digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def _json(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required local input is missing: {path}')
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _environment():
    return dict(python=sys.version, packages={name: version(name) for name in
        ('torch', 'torchvision', 'monai', 'numpy', 'scipy', 'h5py', 'scikit-image', 'scikit-learn')})


def _verified_inputs(args, package_root):
    checkpoint = args.checkpoint.expanduser().resolve()
    root = args.project_root.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Required checkpoint is missing: {checkpoint}')
    if not root.is_dir():
        raise FileNotFoundError(f'Private project root is missing: {root}')
    output = (args.output.expanduser().resolve() if args.output else checkpoint.parent / 'evaluation.json')
    if output.exists():
        raise FileExistsError(f'Preserve existing output; choose a new --output path: {output}')
    if output.suffix.lower() != '.json':
        raise ValueError('--output must be a new .json file')
    vast = package_root.resolve() / 'VAST'
    prepared = vast / 'prepared'
    protocol_path = prepared / 'protocol.json'
    valid_path = prepared / 'valid.json'
    protocol = _json(protocol_path)
    manifest = _json(checkpoint.parent / 'manifest.json')
    if not isinstance(protocol, dict) or not isinstance(manifest, dict):
        raise ValueError('Protocol and manifest must be JSON objects')
    protocol_hash = _digest(protocol_path)
    if manifest.get('protocol_sha256') != protocol_hash:
        raise ValueError('Manifest protocol digest mismatch; use the original prepared protocol')
    if Path(protocol.get('project_root', '')).resolve() != root:
        raise ValueError('--project-root differs from the prepared protocol; use the original private workspace')
    if protocol.get('validation_policy') != EVALUATION_POLICY or manifest.get('evaluation_policy') != EVALUATION_POLICY:
        raise ValueError('Unsupported evaluation policy; this command requires the fixed known-view protocol')
    if protocol.get('environment') != _environment():
        raise ValueError('Python/package environment differs from the prepared protocol')
    sources = protocol.get('source_hashes', {})
    if not isinstance(sources, dict) or not all(name in sources for name in REQUIRED_SOURCES):
        raise ValueError('Protocol is missing required evaluation source hashes')
    if manifest.get('source_hashes') != sources:
        raise ValueError('Manifest and protocol source hashes disagree')
    for name, expected in sources.items():
        path = (vast / name).resolve()
        if not path.is_relative_to(vast) or not path.is_file() or _digest(path) != expected:
            raise ValueError('Source changed since prepare: ' + str(name))
    valid_hash = protocol.get('split_hashes', {}).get('valid')
    if not valid_path.is_file() or not valid_hash or _digest(valid_path) != valid_hash:
        raise ValueError('Prepared validation split digest mismatch')
    if manifest.get('split_hashes', {}).get('valid') != valid_hash:
        raise ValueError('Manifest validation split digest mismatch')
    expected_ids = protocol.get('ordered_ids', {}).get('valid')
    if not isinstance(expected_ids, list) or not expected_ids or manifest.get('validation_ids') != expected_ids:
        raise ValueError('Manifest and protocol validation order disagree')
    records = _json(valid_path)
    if not isinstance(records, list) or len(records) != len(expected_ids):
        raise ValueError('Validation split count disagrees with protocol')
    if protocol.get('counts', {}).get('valid', {}).get('n') != len(records):
        raise ValueError('Validation count is missing or inconsistent in protocol')
    observed_ids = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Every validation entry must contain image and label paths')
        for key in ('image', 'label'):
            value = record.get(key)
            if not isinstance(value, str) or not value:
                raise ValueError(f'Validation entry is missing a {key} path')
            path = Path(value)
            if not path.is_absolute() or not path.resolve().is_relative_to(root / 'data'):
                raise ValueError(f'Validation {key} must be inside the recorded private data directory')
            if not path.is_file() or path.suffix.lower() not in ('.h5', '.hdf5'):
                raise FileNotFoundError(f'Validation {key} HDF5 input is missing: {path}')
        observed_ids.append(Path(record['image']).stem)
    if observed_ids != expected_ids or len(observed_ids) != len(set(observed_ids)):
        raise ValueError('Prepared validation IDs/order changed')
    best = manifest.get('best_checkpoint')
    checkpoint_hash = _digest(checkpoint)
    if not isinstance(best, dict) or best.get('sha256') != checkpoint_hash:
        raise ValueError('Recorded best checkpoint digest mismatch; only the recorded best checkpoint is supported')
    if manifest.get('smoke') is not False:
        raise ValueError('Use a completed training checkpoint, not a smoke checkpoint')
    if manifest.get('stage') not in range(5) or not isinstance(manifest.get('seed'), int):
        raise ValueError('Manifest stage/seed are invalid')
    saved_args = manifest.get('args')
    required = dict(model='echocare', seg_num_classes=15, cls_num_classes=7,
                    view_num_classes=4, echocare_tuning='lora')
    if not isinstance(saved_args, dict) or any(saved_args.get(k) != v for k, v in required.items()):
        raise ValueError('Unsupported checkpoint model configuration; expected the progressive VAST model')
    return dict(checkpoint=checkpoint, output=output, valid_path=valid_path, manifest=manifest,
                protocol_hash=protocol_hash, checkpoint_hash=checkpoint_hash,
                valid_hash=valid_hash, n_images=len(records), environment=protocol['environment'])


def _load_runtime(package_root):
    vast = package_root.resolve() / 'VAST'
    for path in (vast, vast / 'runtime'):
        sys.path.insert(0, str(path))
    torch = importlib.import_module('torch')
    trainer = importlib.import_module('step_1_unimatch_train')
    dataset = importlib.import_module('dataset.fetus')
    core = importlib.import_module('core')
    return SimpleNamespace(torch=torch, trainer=trainer, dataset_class=dataset.FETUSSemiDataset,
                           loader_class=torch.utils.data.DataLoader, evaluator=core.evaluate)


def run_evaluation(args, *, package_root=PACKAGE_ROOT, runtime=None):
    """Validate provenance, load the complete model strictly, and compare fresh metrics with saved values."""
    inputs = _verified_inputs(args, Path(package_root))
    runtime = runtime or _load_runtime(Path(package_root))
    torch = runtime.torch
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; select --device cpu explicitly for CPU evaluation')
        requested_index = int(args.device.split(':')[1]) if ':' in args.device else 0
        if requested_index >= torch.cuda.device_count():
            raise RuntimeError('Requested CUDA device index is unavailable')
    checkpoint = torch.load(inputs['checkpoint'], map_location='cpu', weights_only=True)
    manifest = inputs['manifest']
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('model'), dict):
        raise ValueError('Expected a progressive checkpoint containing the complete model state')
    if checkpoint.get('protocol_sha256') != inputs['protocol_hash']:
        raise ValueError('Saved checkpoint protocol digest mismatch')
    for key in ('stage', 'seed', 'smoke'):
        if checkpoint.get(key) != manifest[key]:
            raise ValueError(f'Checkpoint and manifest {key} disagree')
    if checkpoint.get('epoch') != manifest['best_checkpoint'].get('epoch'):
        raise ValueError('Checkpoint epoch does not match the recorded best epoch')
    if checkpoint.get('args') != manifest['args']:
        raise ValueError('Checkpoint and manifest model arguments disagree')
    reference = checkpoint.get('validation')
    if not isinstance(reference, dict) or reference.get('n_images') != inputs['n_images']:
        raise ValueError('Checkpoint validation scope does not match the prepared split')
    for key in METRICS:
        saved = reference.get(key)
        if type(saved) not in (int, float) or not math.isfinite(saved):
            raise ValueError('Non-finite or missing saved reference metric: ' + key)
    model_args = argparse.Namespace(**manifest['args'])
    # Full state includes the pretrained encoder: do not load a separate encoder checkpoint.
    model_args.ssl_ckpt = None
    model = runtime.trainer.build_model(model_args, device)
    model.load_state_dict(checkpoint['model'], strict=True)
    dataset = runtime.dataset_class(str(inputs['valid_path']), 'valid')
    loader = runtime.loader_class(dataset, batch_size=1, shuffle=False, num_workers=0)
    seg_allowed = runtime.trainer.build_seg_allowed_mat(device, runtime.trainer.DEFAULT_SEG_ALLOWED, 4, 15)
    metrics = runtime.evaluator(model_args, model, device, loader, seg_allowed, runtime.trainer.DEFAULT_CLS_ALLOWED)
    if metrics.get('n_images') != inputs['n_images']:
        raise ValueError('Evaluation did not cover the entire prepared validation split')
    for key in METRICS:
        value = metrics.get(key)
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError('Non-finite or missing evaluation metric: ' + key)
    differences = {key: metrics[key] - reference[key] for key in METRICS}
    comparison = dict(matches_saved_reference=all(abs(value) <= 1e-5 for value in differences.values()),
                      absolute_tolerance=1e-5,
                      per_metric_differences=differences,
                      difference_definition='fresh evaluation minus saved checkpoint reference',
                      saved_metrics={key: reference[key] for key in METRICS})
    result = dict(evaluation_scope='internal_validation_known_view',
                  evaluation_policy=EVALUATION_POLICY,
                  checkpoint_sha256=inputs['checkpoint_hash'], protocol_sha256=inputs['protocol_hash'],
                  validation_split_sha256=inputs['valid_hash'], stage=manifest['stage'], seed=manifest['seed'],
                  checkpoint_epoch=checkpoint['epoch'], device=str(args.device),
                  environment=inputs['environment'],
                  metric_units={'Dice': 'percent', 'NSD': 'percent', 'Macro-F1': 'fraction'},
                  classification_threshold=0.5, nsd_tolerance_pixels=2.0,
                  saved_reference_comparison=comparison, metrics=metrics)
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    inputs['output'].parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves any existing file even if a second process races this one.
    with inputs['output'].open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(serialized)
    return result


def main(argv=None):
    args = parse_args(argv)
    try:
        result = run_evaluation(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print('Evaluation failed: ' + str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: result['metrics'][key] for key in METRICS}, indent=2))
    if not result['saved_reference_comparison']['matches_saved_reference']:
        print('Saved-reference metrics differ; see signed differences in the output JSON.')
    output = args.output or args.checkpoint.parent / 'evaluation.json'
    print('Saved internal validation evaluation to ' + str(output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
