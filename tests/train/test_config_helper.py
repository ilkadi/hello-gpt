import unittest
from hello_gpt.train.config_helper import ConfigHelper

class TestConfigHelper(unittest.TestCase):
    def setUp(self):
        self.config_helper = ConfigHelper()

    def test_update_with_config(self):
        config = {'key1': 'value1', 'key3': 'value3'}
        config_path = 'tests/train/resources/test_config.yaml'
        expected_config = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}

        updated_config = self.config_helper.update_with_config(config, config_path)

        self.assertEqual(updated_config, expected_config)

    def test_print_config(self):
        config = {'key1': 'value1', 'key2': 'value2'}
        expected_output = "Config:\n{'key1': 'value1', 'key2': 'value2'}\n"

        # Redirect stdout to capture the print output
        from io import StringIO
        import sys
        captured_output = StringIO()
        sys.stdout = captured_output

        self.config_helper.print_config(config)

        # Reset stdout
        sys.stdout = sys.__stdout__

        self.assertEqual(captured_output.getvalue(), expected_output)

if __name__ == '__main__':
    unittest.main()