import unittest
from aad.blocks import apply_blocks,extract

class CppBlockTests(unittest.TestCase):
    def test_cpp_block_preserves_fixed_score(self):
        code='double score(){return 1;}\n// AAD-BEGIN proposal\nint move(){return 1;}\n// AAD-END proposal\n'
        patched=apply_blocks(code,{'proposal':'int move(){return 2;}'},language='cpp')
        self.assertTrue(patched.startswith('double score(){return 1;}'))
        self.assertIn('return 2',extract(patched)['proposal'])
    def test_cpp_marker_injection_rejected(self):
        with self.assertRaises(ValueError):
            apply_blocks('// AAD-BEGIN proposal\nx\n// AAD-END proposal\n',{'proposal':'// AAD-BEGIN escape'},language='cpp')

if __name__=='__main__':unittest.main()
