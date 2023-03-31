import unittest
import importlib

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from modules import processing
from scripts import controlnet, external_code

class TestControlNetScriptWorking(unittest.TestCase):
    def setUp(self):
        self.script = controlnet.Script()

    def test_empty_controlmaps_before_process_batch_does_not_crash(self):
        p = processing.StableDiffusionProcessingTxt2Img()
        self.script.before_process_batch(p, batch_number=1)
