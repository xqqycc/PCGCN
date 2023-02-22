import os.path


class ModelConf(object):
    def __init__(self,file):
        self.config = {}
        self.read_configuration(file)

    def __getitem__(self, item):
        if not self.contain(item):
            print('parameter '+item+' is not found in the configuration file!')
            exit(-1)
        return self.config[item]

    def contain(self,key):#判断是否有相关参数在conf中
        return key in self.config

    def read_configuration(self,file):
        if not os.path.exists(file):#路径上不存在conf文件
            print('config file is not found!')
            raise IOError
        with open(file) as f:
            for ind,line in enumerate(f):#ind表示是哪一行的参数不对
                if line.strip()!='':#按空格划分删除后如若不为空
                    try:
                        key,value=line.strip().split('=')#以等号为划分切割出key和value
                        self.config[key]=value
                    except ValueError:
                        print('config file is not in the correct format! Error Line:%d' % ind)


class OptionConf(object):
    def __init__(self,content):
        self.line = content.strip().split(' ')#读conf中item.ranking那行，按空格划，列表形式-['-topN', '10,20']
        self.options = {}
        self.mainOption = False
        if self.line[0] == 'on':#不会执行
            self.mainOption = True
        elif self.line[0] == 'off':#不会执行
            self.mainOption = False
        for i,item in enumerate(self.line):
            if (item.startswith('-') or item.startswith('--')) and  not item[1:].isdigit():#startswith检查字符串开头，isdigit检查是否全为数字
                ind = i+1
                for j,sub in enumerate(self.line[ind:]):
                    if (sub.startswith('-') or sub.startswith('--')) and  not sub[1:].isdigit():
                        ind = j
                        break
                    if j == len(self.line[ind:])-1:
                        ind=j+1
                        break
                try:
                    self.options[item] = ' '.join(self.line[i+1:i+1+ind])
                except IndexError:
                    self.options[item] = 1

    def __getitem__(self, item):
        if not self.contain(item):
            print('parameter '+item+' is invalid!')
            exit(-1)
        return self.options[item]

    def keys(self):
        return self.options.keys()

    def is_main_on(self):
        return self.mainOption

    def contain(self,key):
        return key in self.options


