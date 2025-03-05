from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, ConditionalContainer
from prompt_toolkit.widgets import Frame, TextArea, RadioList, Box, Label
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.dimension import D
import yaml

class ConfigEditor:
    def __init__(self, config: str, model: str):
        self.model = model 
        self.config = self._load_yaml(config)
        self.config_path = config
        
        self.editable_fields = {}
        self.bindings = self.get_bindings()
        self.setup_ui()
        
        self.index = 0
        
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
                    event.app.layout.focus(fields[self.index])
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
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, f, sort_keys=False)
        
        return bindings
        
    def setup_ui(self):
        self.options = RadioList(values=[(key, key) for key in self.config.keys()])
    
        module_windows = []
        for module in self.config.keys():
            module_windows.append(self.get_module_window(self.options, self.config, module, "GENITEXT"))
        
        options_list = Frame(Box(self.options), title=self.model.upper(), width=35)
        
        layout = Layout(
            HSplit([
                VSplit([options_list] + module_windows),
            ])
        )
        
        self.app = Application(layout=layout, key_bindings=self.bindings, full_screen=True)
        
    def get_module_window(self, options, config: dict, module: str, title: str): 
        self.editable_fields[module] = {}
        field_widgets = [] 
        
        max_label_width = max(len(key) for key in config[module].keys()) + 2
        
        for key, value in config[module].items(): 
            text_field = TextArea(text=str(value), multiline=False)
            def on_change(buffer, key=key):
                config[module][key] = buffer.text
            text_field.buffer.on_text_changed += on_change
            
            self.editable_fields[module][key] = text_field
            
            field_widgets.append(
                VSplit([
                    Label(text=f"{key}: ", width=D.exact(max_label_width)), 
                    text_field
                ], padding=1)
            )
        
        module_container = HSplit(field_widgets, padding=1)
        
        return ConditionalContainer(
            Frame(
                module_container, 
                title=title, 
                width=D(preferred=60)
            ), 
            filter=Condition(lambda: options.current_value == module)
        )
        
    def run(self):
        self.app.run()