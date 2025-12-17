# ape/template.py

class EvalTemplate:
    """
    負責處理評估時的 Prompt 填空。
    格式支援: [PROMPT], [INPUT], [OUTPUT], [full_DEMO]
    """
    def __init__(self, template):
        self.template = template

    def fill(self, prompt='', full_demo='', input='', output=''):
        return self.template.replace('[PROMPT]', prompt)\
                            .replace('[full_DEMO]', full_demo)\
                            .replace('[INPUT]', input)\
                            .replace('[OUTPUT]', output)

class DemosTemplate:
    """
    負責處理 Few-shot 範例的填空。
    格式支援: [INPUT], [OUTPUT]
    """
    def __init__(self, template, delimiter='\n\n'):
        self.template = template
        self.delimiter = delimiter

    def fill(self, data):
        """
        Data 格式應為 tuple: (inputs_list, outputs_list)
        """
        if not data or not data[0]:
            return ""
            
        inputs, outputs = data
        demos = []
        for input_, output_ in zip(inputs, outputs):
            # 處理 output 可能為 list 的情況 (如 instruction induction dataset)
            out_str = output_[0] if isinstance(output_, list) else output_
            
            demo = self.template.replace('[INPUT]', input_)\
                                .replace('[OUTPUT]', out_str)
            demos.append(demo)
            
        return self.delimiter.join(demos)