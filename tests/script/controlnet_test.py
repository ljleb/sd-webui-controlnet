import unittest
import importlib

utils = importlib.import_module('extensions.sd-webui-controlnet.tests.utils', 'utils')
utils.setup_test_env()

from modules import processing
from scripts import controlnet, external_code

class TestControlNetScriptWorking(unittest.TestCase):
    def get_dummy_image(self, i=0):
        return f'base64#{i}...'

    def setUp(self):
        self.script = controlnet.Script()
        self.enabled_units = []

    def assert_normalize_to_batch_mode_works(self, batch_images_list):
        units, batch_size = self.script.normalize_to_batch_mode(self.enabled_units)

        batch_units = [unit for unit in self.enabled_units if getattr(unit, 'input_mode', controlnet.InputMode.SIMPLE) == controlnet.InputMode.BATCH]
        if batch_units:
            self.assertEqual(min(len(unit.batch_images) for unit in batch_units), batch_size)
        else:
            self.assertEqual(1, batch_size)

        for i, unit in enumerate(units):
            self.assertEqual(controlnet.InputMode.BATCH, unit.input_mode)
            self.assertEqual(batch_images_list[i], unit.batch_images)

    def test_empty_controlmaps_before_process_batch_does_not_crash(self):
        p = processing.StableDiffusionProcessingTxt2Img()
        self.script.before_process_batch(p, batch_number=1)

    def test_normalize_to_batch_mode_empty(self):
        units, batch_size = self.script.normalize_to_batch_mode([])
        self.assertEqual(units, [])
        self.assertEqual(batch_size, 1)

    def test_normalize_to_batch_mode_1_simple(self):
        self.enabled_units.append(external_code.ControlNetUnit(image=self.get_dummy_image()))
        self.assert_normalize_to_batch_mode_works([
            [self.enabled_units[0].image],
        ])

    def test_normalize_to_batch_mode_2_simples(self):
        self.enabled_units.extend([
            external_code.ControlNetUnit(image=self.get_dummy_image(0)),
            external_code.ControlNetUnit(image=self.get_dummy_image(1)),
        ])
        self.assert_normalize_to_batch_mode_works([
            [self.get_dummy_image(0)],
            [self.get_dummy_image(1)],
        ])

    def test_normalize_to_batch_mode_1_batch(self):
        self.enabled_units.extend([
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(0),
                    self.get_dummy_image(1),
                ],
            ),
        ])
        self.assert_normalize_to_batch_mode_works([
            [
                self.get_dummy_image(0),
                self.get_dummy_image(1),
            ],
        ])

    def test_normalize_to_batch_mode_2_batches(self):
        self.enabled_units.extend([
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(0),
                    self.get_dummy_image(1),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(2),
                    self.get_dummy_image(3),
                ],
            ),
        ])
        self.assert_normalize_to_batch_mode_works([
            [
                self.get_dummy_image(0),
                self.get_dummy_image(1),
            ],
            [
                self.get_dummy_image(2),
                self.get_dummy_image(3),
            ],
        ])

    def test_normalize_to_batch_mode_2_mixed(self):
        self.enabled_units.extend([
            external_code.ControlNetUnit(image=self.get_dummy_image(0)),
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(1),
                    self.get_dummy_image(2),
                ],
            ),
        ])
        self.assert_normalize_to_batch_mode_works([
            [
                self.get_dummy_image(0),
            ],
            [
                self.get_dummy_image(1),
                self.get_dummy_image(2),
            ],
        ])

    def test_normalize_to_batch_mode_3_mixed(self):
        self.enabled_units.extend([
            external_code.ControlNetUnit(image=self.get_dummy_image(0)),
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(1),
                    self.get_dummy_image(2),
                    self.get_dummy_image(3),
                ],
            ),
            controlnet.UiControlNetUnit(
                input_mode=controlnet.InputMode.BATCH,
                batch_images=[
                    self.get_dummy_image(4),
                    self.get_dummy_image(5),
                ],
            ),
        ])
        self.assert_normalize_to_batch_mode_works([
            [
                self.get_dummy_image(0),
            ],
            [
                self.get_dummy_image(1),
                self.get_dummy_image(2),
            ],
            [
                self.get_dummy_image(4),
                self.get_dummy_image(5),
            ],
        ])
