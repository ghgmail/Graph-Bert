'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from code.base_class.setting import setting
import pandas as pd

class Settings(setting):
    fold = None
    
    def load_run_save_evaluate(self):
        
        # load dataset
        loaded_data = self.dataset.load()
        # pd.to_pickle(loaded_data,'./result/preprocess/data_preprocess.pkl')
        print("把预处理的数据存到本地result/cora_pre/cora_pre.pkl")
        pd.to_pickle(loaded_data,"/mnt/d/zk_files/dd/project/Graph-Bert/result/cora_pre/cora_pre.pkl")
        print("保存完成！")

        # run learning methods
        self.method.data = loaded_data
        learned_result = self.method.run()
            
        # save raw ResultModule
        self.result.data = learned_result
        self.result.save()

        # evaluate learning results
        if self.evaluate is not None:
            self.evaluate.data = learned_result
            self.evaluate.evaluate()

        return None

        