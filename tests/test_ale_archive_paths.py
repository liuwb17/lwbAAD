import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

from aad.ale_batch import archive_files
from aad.ale_evaluation import prepare, load_task
import test_parallel_engine as fixtures


class ArchivePathsTests(unittest.TestCase):
    def test_official_double_separator_preserves_public_inputs_and_tool_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = fixtures.AleBundleTests().archive(folder, kind='reactive')
            rewritten = Path(folder) / 'official.zip'
            with zipfile.ZipFile(archive) as original, zipfile.ZipFile(rewritten, 'w') as output:
                for item in original.infolist():
                    output.writestr(item.filename.replace('toy/', 'toy//', 1), original.read(item))
            bundle = Path(folder) / 'bundle'
            with patch('aad.ale_batch.build_tools'), patch('aad.ale_batch.generate',
                    side_effect=lambda seeds, _: [f'input {s}' for s in seeds]):
                prepare(rewritten, 'toy', bundle)
            task, cases = load_task(bundle)
            self.assertEqual(task['problem_type'], 'reactive')
            self.assertEqual([row['seed'] for row in task['cases']], list(range(20)))
            self.assertEqual(len(cases), 20)
            self.assertTrue(task['tool_sources'])

    def test_ambiguous_or_traversal_members_are_rejected(self):
        for names in [('toy/data.json', 'toy//data.json'), ('toy/../data.json',)]:
            with self.subTest(names=names), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'bad.zip'
                with zipfile.ZipFile(path, 'w') as output:
                    for name in names:
                        output.writestr(name, '{}')
                with zipfile.ZipFile(path) as source, self.assertRaises(ValueError):
                    archive_files(source)
