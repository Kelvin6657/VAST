"""Synthetic evaluation CLI tests; no study data or checkpoint is required."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'eval' / 'evaluate.py'


def load_module():
    spec = importlib.util.spec_from_file_location('vast_evaluate_cli', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE_PATH.is_file(), 'The standalone evaluation CLI is missing')
        self.cli = load_module()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.package = self.base / 'release'
        self.root = self.base / 'private_workspace'
        self.vast = self.package / 'VAST'
        self.prepared = self.vast / 'prepared'
        self.prepared.mkdir(parents=True)
        (self.root / 'data').mkdir(parents=True)
        self.checkpoint = self.base / 'run' / 'best.pth'
        self.checkpoint.parent.mkdir()
        self.checkpoint.write_bytes(b'synthetic checkpoint bytes, not a torch serialization')
        self.source_hashes = {}
        for name in self.cli.REQUIRED_SOURCES:
            path = self.vast / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# synthetic source\n', encoding='utf-8')
            self.source_hashes[name] = self.sha(path)
        image = self.root / 'data' / 'synthetic_image.h5'
        label = self.root / 'data' / 'synthetic_label.h5'
        image.touch()
        label.touch()
        self.write(self.prepared / 'valid.json', [{'image': str(image), 'label': str(label)}])
        self.environment = {'python': 'synthetic interpreter', 'packages': {'torch': 'synthetic'}}
        self.protocol = {
            'project_root': str(self.root.resolve()),
            'validation_policy': self.cli.EVALUATION_POLICY,
            'source_hashes': self.source_hashes,
            'split_hashes': {'valid': self.sha(self.prepared / 'valid.json')},
            'ordered_ids': {'valid': ['synthetic_image']},
            'counts': {'valid': {'n': 1}},
            'environment': self.environment,
        }
        self.write(self.prepared / 'protocol.json', self.protocol)
        self.protocol_hash = self.sha(self.prepared / 'protocol.json')
        self.model_args = dict(model='echocare', seg_num_classes=15, cls_num_classes=7,
                               view_num_classes=4, echocare_tuning='lora', ssl_ckpt='unused.pth')
        self.manifest = dict(stage=4, seed=42, smoke=False, args=self.model_args,
                             protocol_sha256=self.protocol_hash,
                             evaluation_policy=self.cli.EVALUATION_POLICY,
                             source_hashes=self.source_hashes,
                             split_hashes=self.protocol['split_hashes'],
                             validation_ids=['synthetic_image'],
                             best_checkpoint={'sha256': self.sha(self.checkpoint), 'epoch': 7})
        self.write(self.checkpoint.parent / 'manifest.json', self.manifest)
        self.args = argparse.Namespace(checkpoint=self.checkpoint, project_root=self.root,
                                       device='cpu', output=None)
        self.checkpoint_value = dict(model={'synthetic_tensor': 1}, stage=4, seed=42,
                                     smoke=False, epoch=7, args=self.model_args,
                                     protocol_sha256=self.protocol_hash,
                                     validation=dict(Dice=10.0, NSD=5.0, **{'Macro-F1': 0.2},
                                                     score=27.5, n_images=1))

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value), encoding='utf-8')

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_cli(self, runtime=None):
        with patch.object(self.cli, '_environment', return_value=self.environment):
            return self.cli.run_evaluation(self.args, package_root=self.package, runtime=runtime)

    def runtime(self):
        model = Mock()
        trainer = SimpleNamespace(build_model=Mock(return_value=model),
            build_seg_allowed_mat=Mock(return_value='seg-mask'),
            DEFAULT_SEG_ALLOWED='seg-mapping', DEFAULT_CLS_ALLOWED='cls-mapping')
        metrics = dict(self.checkpoint_value['validation'])
        torch = SimpleNamespace(load=Mock(return_value=self.checkpoint_value),
            device=lambda value: SimpleNamespace(type=value.split(':')[0]),
            cuda=SimpleNamespace(is_available=lambda: False, device_count=lambda: 0))
        return SimpleNamespace(torch=torch, trainer=trainer,
            dataset_class=Mock(return_value=['synthetic-item']),
            loader_class=Mock(return_value='loader'), evaluator=Mock(return_value=metrics))

    def test_required_arguments_and_default_device(self):
        with self.assertRaises(SystemExit):
            self.cli.parse_args([])
        args = self.cli.parse_args(['--checkpoint', str(self.checkpoint),
                                    '--project-root', str(self.root)])
        self.assertEqual(args.device, 'cuda:0')

    def test_loads_strictly_and_uses_fixed_validation_loader(self):
        runtime = self.runtime()
        result = self.run_cli(runtime)
        runtime.torch.load.assert_called_once_with(self.checkpoint.resolve(),
                                                   map_location='cpu', weights_only=True)
        built_args = runtime.trainer.build_model.call_args.args[0]
        self.assertIsNone(built_args.ssl_ckpt)
        runtime.trainer.build_model.return_value.load_state_dict.assert_called_once_with(
            self.checkpoint_value['model'], strict=True)
        runtime.loader_class.assert_called_once_with(['synthetic-item'], batch_size=1,
                                                     shuffle=False, num_workers=0)
        self.assertEqual(runtime.evaluator.call_args.args[-1], 'cls-mapping')
        self.assertEqual(result['evaluation_scope'], 'internal_validation_known_view')
        self.assertEqual(result['metrics']['n_images'], 1)
        self.assertTrue(result['saved_reference_comparison']['matches_saved_reference'])
        self.assertEqual(json.loads((self.checkpoint.parent / 'evaluation.json').read_text()), result)
        self.assertNotIn('validation_ids', result)

    def test_missing_checkpoint_fails_before_loading_runtime(self):
        self.checkpoint.unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'checkpoint'):
            self.run_cli(self.runtime())

    def test_modified_checkpoint_is_rejected_by_recorded_digest(self):
        self.checkpoint.write_bytes(b'modified synthetic bytes')
        with self.assertRaisesRegex(ValueError, 'checkpoint.*digest'):
            self.run_cli(self.runtime())

    def test_modified_validation_manifest_is_rejected(self):
        (self.prepared / 'valid.json').write_text('[]')
        with self.assertRaisesRegex(ValueError, 'validation split.*digest'):
            self.run_cli(self.runtime())

    def test_modified_source_is_rejected(self):
        (self.vast / 'core.py').write_text('# changed\n')
        with self.assertRaisesRegex(ValueError, 'Source changed'):
            self.run_cli(self.runtime())

    def test_changed_environment_is_rejected(self):
        with patch.object(self.cli, '_environment', return_value={'python': 'different'}):
            with self.assertRaisesRegex(ValueError, 'environment'):
                self.cli.run_evaluation(self.args, package_root=self.package, runtime=self.runtime())

    def test_wrong_checkpoint_protocol_is_rejected_before_model_build(self):
        runtime = self.runtime()
        self.checkpoint_value['protocol_sha256'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'checkpoint protocol'):
            self.run_cli(runtime)
        runtime.trainer.build_model.assert_not_called()

    def test_strict_state_load_failure_does_not_write_results(self):
        runtime = self.runtime()
        runtime.trainer.build_model.return_value.load_state_dict.side_effect = RuntimeError('missing keys')
        with self.assertRaisesRegex(RuntimeError, 'missing keys'):
            self.run_cli(runtime)
        runtime.evaluator.assert_not_called()
        self.assertFalse((self.checkpoint.parent / 'evaluation.json').exists())

    def test_valid_changed_metrics_are_saved_with_reference_mismatch(self):
        runtime = self.runtime()
        runtime.evaluator.return_value['Dice'] = 99.0
        result = self.run_cli(runtime)
        comparison = result['saved_reference_comparison']
        self.assertFalse(comparison['matches_saved_reference'])
        self.assertEqual(comparison['absolute_tolerance'], 1e-5)
        self.assertEqual(comparison['per_metric_differences']['Dice'], 89.0)
        self.assertEqual(comparison['saved_metrics']['Dice'], 10.0)
        saved = json.loads((self.checkpoint.parent / 'evaluation.json').read_text())
        self.assertEqual(saved['metrics']['Dice'], 99.0)

    def test_nonfinite_saved_reference_is_rejected(self):
        runtime = self.runtime()
        self.checkpoint_value['validation']['Dice'] = float('inf')
        with self.assertRaisesRegex(ValueError, 'saved reference.*Dice'):
            self.run_cli(runtime)
        self.assertFalse((self.checkpoint.parent / 'evaluation.json').exists())

    def test_missing_saved_reference_metric_is_rejected(self):
        runtime = self.runtime()
        del self.checkpoint_value['validation']['NSD']
        with self.assertRaisesRegex(ValueError, 'saved reference.*NSD'):
            self.run_cli(runtime)
        self.assertFalse((self.checkpoint.parent / 'evaluation.json').exists())

    def test_cuda_unavailable_fails_without_cpu_fallback(self):
        self.args.device = 'cuda:0'
        runtime = self.runtime()
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            self.run_cli(runtime)
        runtime.trainer.build_model.assert_not_called()

    def test_existing_output_is_preserved(self):
        path = self.checkpoint.parent / 'evaluation.json'
        path.write_text('preserve this existing file')
        with self.assertRaises(FileExistsError):
            self.run_cli(self.runtime())
        self.assertEqual(path.read_text(), 'preserve this existing file')


    def test_manifest_policy_cannot_silently_change_view_handling(self):
        self.manifest['evaluation_policy'] = 'unmasked prediction'
        self.write(self.checkpoint.parent / 'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'evaluation policy'):
            self.run_cli(self.runtime())

    def test_wrong_private_root_is_rejected(self):
        self.args.project_root = self.root / 'other_workspace'
        self.args.project_root.mkdir()
        with self.assertRaisesRegex(ValueError, 'project-root differs'):
            self.run_cli(self.runtime())

    def test_unavailable_validation_file_is_rejected_before_loading_model(self):
        (self.root / 'data' / 'synthetic_image.h5').unlink()
        runtime = self.runtime()
        with self.assertRaisesRegex(FileNotFoundError, 'HDF5 input is missing'):
            self.run_cli(runtime)
        runtime.torch.load.assert_not_called()

    def test_custom_output_is_used_without_modifying_manifest(self):
        self.args.output = self.base / 'evaluations' / 'reproduced.json'
        before = (self.checkpoint.parent / 'manifest.json').read_bytes()
        self.run_cli(self.runtime())
        self.assertTrue(self.args.output.is_file())
        self.assertFalse((self.checkpoint.parent / 'evaluation.json').exists())
        self.assertEqual((self.checkpoint.parent / 'manifest.json').read_bytes(), before)

    def test_smoke_checkpoint_is_rejected(self):
        self.manifest['smoke'] = True
        self.write(self.checkpoint.parent / 'manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'smoke checkpoint'):
            self.run_cli(self.runtime())

    def test_nonfinite_metric_cannot_be_saved(self):
        runtime = self.runtime()
        runtime.evaluator.return_value['Dice'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            self.run_cli(runtime)
        self.assertFalse((self.checkpoint.parent / 'evaluation.json').exists())

if __name__ == '__main__':
    unittest.main()

