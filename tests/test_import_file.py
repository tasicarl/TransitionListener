"""A model file loaded by path can import the modules beside it, from any working directory."""
import os
import sys
import tempfile
import textwrap
import unittest

from transitionlistener.helper_functions import import_file


class ImportFileTests(unittest.TestCase):
    def test_a_model_imports_its_sibling_from_another_directory(self):
        with tempfile.TemporaryDirectory() as models, tempfile.TemporaryDirectory() as elsewhere:
            with open(os.path.join(models, "tl_test_parent_model.py"), "w") as f:
                f.write("VALUE = 42\n")
            with open(os.path.join(models, "tl_test_child_model.py"), "w") as f:
                f.write(textwrap.dedent("""\
                    import tl_test_parent_model as parent
                    VALUE = parent.VALUE + 1
                """))
            cwd, path = os.getcwd(), list(sys.path)
            try:
                os.chdir(elsewhere)
                module = import_file(os.path.join(models, "tl_test_child_model.py"))
                self.assertEqual(module.VALUE, 43)
            finally:
                os.chdir(cwd)
                sys.path[:] = path
                sys.modules.pop("tl_test_parent_model", None)


if __name__ == "__main__":
    unittest.main()
