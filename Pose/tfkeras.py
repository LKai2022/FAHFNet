from utils import inject_tfkeras_modules, init_tfkeras_custom_objects
import freqfusion as model

FreqFusion = inject_tfkeras_modules(model.FreqFusion)

preprocess_input = inject_tfkeras_modules(model.preprocess_input)

init_tfkeras_custom_objects()
