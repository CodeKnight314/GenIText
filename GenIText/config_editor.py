from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, ConditionalContainer, Window, FormattedTextControl
from prompt_toolkit.widgets import Frame, TextArea, RadioList, Box, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.dimension import D
from prompt_toolkit.formatted_text import FormattedText
import yaml

class ConfigEditor:
    def __init__(self, config: str, model: str):
        self.model = model 
        self.config = self._load_yaml(config)
        self.config_path = config
        
        self.editable_fields = {}
        self.error_labels = {}
        
        self.bindings = self.get_bindings()
        self.setup_ui()
        
        self.model_urls = {
            "llava": ["llava-hf/llava-1.5-7b-hf", 
                      "llava-hf/llava-1.5-13b-hf"],
            "vit_gpt2": ["nlpconnect/vit-gpt2-image-captioning", 
                         "ifmain/vit-gpt2-image2promt-stable-diffusion",
                         "ashok2216/vit-gpt2-image-captioning_COCO_FineTuned"], 
            "blipv2": ["Salesforce/blip2-opt-2.7b",
                       "Salesforce/blip2-opt-6.7b",
                       "Salesforce/blip2-opt-6.7b-coco",
                       "Salesforce/blip2-flan-t5-xxl",
                       "Salesforce/blip2-flan-t5-xl"]
        }
        
    def _load_yaml(self, path: str): 
        with open(path, 'r') as f: 
            return yaml.safe_load(f)
    
    def get_bindings(self):
        bindings = KeyBindings()
        
        @bindings.add('c-c')
        def exit_(event):
            event.app.exit()
        
        @bindings.add('tab')
        def switch_focus(event):
            module = self.options.current_value
            fields = list(self.editable_fields.get(module, {}).values())
            if event.app.layout.has_focus(self.options):
                if fields:
                    event.app.layout.focus(fields[0])
                return
            else: 
                event.app.layout.focus(self.options)
                
        @bindings.add('up')
        def focus_up(event):
            module = self.options.current_value
            fields = list(self.editable_fields.get(module, {}).values())
            current_control = event.app.layout.current_control
            
            if event.app.layout.focus(self.options):
                return
            
            idx = None
            for i, field in enumerate(fields):
                if field.control == current_control: 
                    idx = i
                    break
            
            if idx is None:
                idx = 0
            
            idx = idx - 1
            if idx < 0: 
                event.app.layout.focus(fields[-1])
            else: 
                event.app.layout.focus(fields[idx])
        
        @bindings.add("down")
        def focus_down(event):
            module = self.options.current_value
            fields = list(self.editable_fields.get(module, {}).values())
            current_control = event.app.layout.current_control
            
            if event.app.layout.focus(self.options):
                return
            
            idx = None
            for i, field in enumerate(fields):
                if field.control == current_control: 
                    idx = i
                    break
            
            if idx is None:
                idx = 0
            
            idx = idx + 1
            if idx >= len(fields): 
                event.app.layout.focus(fields[0])
            else: 
                event.app.layout.focus(fields[idx])
                
        @bindings.add("c-s")
        def save_config(event):
            for module, fields in self.editable_fields.items():
                for key, field in fields.items():
                    try:
                        self.config[module][key] = yaml.safe_load(field.text)
                    except Exception:
                        self.config[module][key] = field.text
            
            model_errors = self.assert_model_edits()
            gen_errors = self.assert_generation_edits()
            proc_errors = self.assert_processor_edits()
            
            all_errors = model_errors + gen_errors + proc_errors
            
            for module in self.error_labels: 
                for key in self.error_labels[module]:
                    self.error_labels[module][key].text = ""
                    
            if all_errors:
                for module, field, message in all_errors:
                    if module in self.error_labels and field in self.error_labels[module]:
                        if self.error_labels[module][field].text == "":
                            self.error_labels[module][field].text = message
            else:
                with open(self.config_path, 'w') as f:
                    yaml.dump(self.config, f, sort_keys=False)
        
        return bindings
        
    def setup_ui(self):
        self.options = RadioList(values=[(key, key) for key in self.config.keys()])
    
        module_windows = []
        for module in self.config.keys():
            module_windows.append(self.get_module_window(self.options, self.config, module, "GENITEXT"))
        
        options_list = Frame(Box(self.options), title=self.model.upper(), width=35, height=D(min=25))
        
        help_section = Frame(Window(content=FormattedTextControl(FormattedText([  
            ("class:red", " ██████╗ ███████╗███╗   ██╗██╗████████╗███████╗██╗  ██╗████████╗\n"),          
            ("class:red", "██╔════╝ ██╔════╝████╗  ██║██║╚══██╔══╝██╔════╝╚██╗██╔╝╚══██╔══╝\n"),
            ("class:red", "██║  ███╗█████╗  ██╔██╗ ██║██║   ██║   █████╗   ╚███╔╝    ██║   \n"),
            ("class:red", "██║   ██║██╔══╝  ██║╚██╗██║██║   ██║   ██╔══╝   ██╔██╗    ██║   \n"),
            ("class:red", "╚██████╔╝███████╗██║ ╚████║██║   ██║   ███████╗██╔╝ ██╗   ██║   \n"),
            ("class:red", " ╚═════╝ ╚══════╝╚═╝  ╚═══╝╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝   ╚═╝   \n"),
            ("class:bold", "---------------------------------------------------------------\n"),
            ("class:bold", "Ctrl + S to save your changes."), 
            ("class:bold", "\nCtrl + C to quit the interface"),
            ("class:bold", "\nEnter to select which module to modify"),
            ("class:bold", "\nUp and down arrow keys to traverse configs"),
            ("class:bold", "\nTab to switch between module and config section")]
        ))), height=D(preferred=15))
        
        layout = Layout(
            HSplit([
                help_section,
                VSplit([options_list] + module_windows)
            ])
        )
        
        self.app = Application(layout=layout, key_bindings=self.bindings, full_screen=True)
        
    def get_module_window(self, options, config: dict, module: str, title: str): 
        self.editable_fields[module] = {}
        self.error_labels.setdefault(module, {})
        field_widgets = [] 

        max_label_width = max(len(key) for key in config[module].keys()) + 2

        for key, value in config[module].items(): 
            text_field = TextArea(text=str(value), multiline=False)
            error_label = Label(text="", style="class:error")  # initially empty

            self.editable_fields[module][key] = text_field
            self.error_labels[module][key] = error_label

            field_widgets.append(
                VSplit([
                    Label(text=f"{key}: ", width=D.exact(max_label_width)), 
                    text_field,
                    error_label
                ], padding=1)
            )

        module_container = HSplit(field_widgets, padding=1)

        return ConditionalContainer(
            Frame(
                module_container, 
                title=title, 
                width=D(preferred=60),
                height=D(min=25)
            ), 
            filter=Condition(lambda: options.current_value == module)
        )

    def run(self):
        self.app.run()
        
    def assert_model_edits(self):
        model_config = self.config["model"]

        validations = [
            ("model_id", model_config["model_id"] in self.model_urls[self.model],
            "model_id is not in the list of valid URLs for this model."),
            ("device", model_config["device"] in {"auto", "mps", "cuda", "cpu"},
            "device must be one of: auto, mps, cuda, cpu."),
            ("dtype", model_config["dtype"] in {"float16", "float32"},
            "dtype must be either 'float16' or 'float32'."),
            ("low_cpu_mem", isinstance(model_config.get("low_cpu_mem"), bool),
            "low_cpu_mem must be a boolean."),
            ("auto_batch", isinstance(model_config.get("auto_batch"), bool),
            "auto_batch must be a boolean."),
            ("quantize.enabled", isinstance(model_config["quantize"].get("enabled"), bool),
            "quantize.enabled must be a boolean.")
        ]

        incorrect_edits = []
        for field, valid, message in validations:
            if not valid:
                incorrect_edits.append(("model", field, message))

        if model_config["quantize"]["enabled"]:
            if model_config["quantize"].get("quant_type") not in {"4bit", "8bit"}:
                incorrect_edits.append(("model", "quantize.quant_type", "quantize.quant_type must be '4bit' or '8bit'."))

        return incorrect_edits

    def assert_generation_edits(self):
        generation_config = self.config["generation"]

        validations = [
            ("min_new_tokens", isinstance(generation_config["min_new_tokens"], int),
            "min_new_tokens must be an integer."),
            ("min_new_tokens", isinstance(generation_config["min_new_tokens"], int) and generation_config["min_new_tokens"] >= 0,
            "min_new_tokens must be non-negative."),

            ("max_new_tokens", isinstance(generation_config["max_new_tokens"], int),
            "max_new_tokens must be an integer."),
            ("max_new_tokens", isinstance(generation_config["max_new_tokens"], int) and generation_config["max_new_tokens"] > 0,
            "max_new_tokens must be greater than 0."),
            ("max_new_tokens", (isinstance(generation_config["max_new_tokens"], int) and 
                                isinstance(generation_config["min_new_tokens"], int) and 
                                generation_config["max_new_tokens"] > generation_config["min_new_tokens"]),
             
            "max_new_tokens must be greater than min_new_tokens."),

            ("num_beams", isinstance(generation_config["num_beams"], int),
            "num_beams must be an integer."),
            ("num_beams", isinstance(generation_config["num_beams"], int) and generation_config["num_beams"] > 0,
            "num_beams must be greater than 0."),

            ("do_sample", isinstance(generation_config["do_sample"], bool),
            "do_sample must be a boolean."),

            ("temperature", isinstance(generation_config["temperature"], (int, float)),
            "temperature must be a number."),
            ("temperature", isinstance(generation_config["temperature"], (int, float)) and 0 <= generation_config["temperature"] <= 1.0,
            "temperature must be between 0 and 1.0."),

            ("top_k", isinstance(generation_config["top_k"], int),
            "top_k must be an integer."),
            ("top_k", isinstance(generation_config["top_k"], int) and generation_config["top_k"] >= 1,
            "top_k must be at least 1."),

            ("top_p", isinstance(generation_config["top_p"], (int, float)),
            "top_p must be a number."),
            ("top_p", isinstance(generation_config["top_p"], (int, float)) and 0 <= generation_config["top_p"] <= 1.0,
            "top_p must be between 0 and 1.0."),

            ("repetition_penalty", isinstance(generation_config["repetition_penalty"], (int, float)),
            "repetition_penalty must be a number."),
            ("repetition_penalty", isinstance(generation_config["repetition_penalty"], (int, float)) and generation_config["repetition_penalty"] >= 0,
            "repetition_penalty must be non-negative."),

            ("length_penalty", isinstance(generation_config["length_penalty"], (int, float)),
            "length_penalty must be a number."),
            ("length_penalty", isinstance(generation_config["length_penalty"], (int, float)) and generation_config["length_penalty"] >= 0,
            "length_penalty must be non-negative."),

            ("no_repeat_ngram_size", isinstance(generation_config["no_repeat_ngram_size"], int),
            "no_repeat_ngram_size must be an integer."),
            ("no_repeat_ngram_size", isinstance(generation_config["no_repeat_ngram_size"], int) and generation_config["no_repeat_ngram_size"] > 0,
            "no_repeat_ngram_size must be greater than 0."),

            ("early_stopping", isinstance(generation_config["early_stopping"], bool),
            "early_stopping must be a boolean."),

            ("return_dict_in_generate", isinstance(generation_config["return_dict_in_generate"], bool),
            "return_dict_in_generate must be a boolean."),

            ("output_scores", isinstance(generation_config["output_scores"], bool),
            "output_scores must be a boolean.")
        ]

        incorrect_edits = []
        for field, valid, message in validations:
            if not valid:
                incorrect_edits.append(("generation", field, message))

        return incorrect_edits


    def assert_processor_edits(self):
        processor_config = self.config["processor"]

        validations = [
            ("model_id",
            processor_config["model_id"] == self.config["model"]["model_id"] and
            processor_config["model_id"] in self.model_urls[self.model],
            "processor model_id must match the model configuration and be in the valid URLs."),

            ("device",
            processor_config["device"] in {"auto", "mps", "cuda", "cpu"},
            "device must be one of: auto, mps, cuda, cpu."),

            ("return_tensors",
            processor_config["return_tensors"] in ["pt"],
            "return_tensors must be 'pt'."),

            ("padding",
            processor_config["padding"] == "max_length",
            "padding must be set to 'max_length'."),

            ("default_prompt",
            isinstance(processor_config["default_prompt"], str),
            "default_prompt must be a string."),

            ("img_h", isinstance(processor_config["img_h"], int),
            "img_h must be an integer."),
            ("img_h", isinstance(processor_config["img_h"], int) and processor_config["img_h"] > 0,
            "img_h must be greater than 0."),

            ("img_w", isinstance(processor_config["img_w"], int),
            "img_w must be an integer."),
            ("img_w", isinstance(processor_config["img_w"], int) and processor_config["img_w"] > 0,
            "img_w must be greater than 0."),

            ("batch_size", isinstance(processor_config["batch_size"], int),
            "batch_size must be an integer."),
            ("batch_size", isinstance(processor_config["batch_size"], int) and processor_config["batch_size"] > 0,
            "batch_size must be greater than 0.")
        ]

        incorrect_edits = []
        for field, valid, message in validations:
            if not valid:
                incorrect_edits.append(("processor", field, message))

        return incorrect_edits
