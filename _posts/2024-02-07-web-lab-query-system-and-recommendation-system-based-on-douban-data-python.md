---
layout: post
title: Web Lab --- Query System and Recommendation System based on Douban Data(Python)
date: 2024-02-07 12:18 +0800
---
# 实验 1 信息获取与检索分析

时数 PB21000052

洪毓谦 PB21000010

## 第 1 阶段 豆瓣数据的爬取与检索

### 爬虫

#### BeautifulSoup 解析

使用 Python 的 requests 库，对电影和书籍对应 ID 的 URL 内容进行爬取，并使用 BeautifulSoup 对页面进行解析。

```Python
import requests
from bs4 import BeautifulSoup

html = requests.get(href, headers=u_a, cookies=c_d) # 发送请求报文获取返回的html
soup = BeautifulSoup(html.text, 'html.parser')  # 用 html.parser 来解析网页
item = soup.find('div', id='content')

book['书籍名称'] = soup.find('span', property='v:itemreviewed').text
# ...
try:
    try:
        book['书籍简介'] = item.find('span', class_='all hidden').text
    except:
        book['书籍简介'] = item.find('div', class_='related_info').find('div', class_='intro').text
except:
    print("本书无简介！")
```

#### XPath 解析

程序还有一种页面解析方案：使用 XPath 定位元素。下面是利用 lxml 库解析元素的例子：

```Python
from lxml import etree

text = requests.get(href, headers=u_a, cookies=c_d).text()
html = etree.HTML(text)
rating = html.xpath('//*[@id="interest_sectl"]/div[1]/div[2]/strong/text()')
new_rating = ["".join(s.split()) for s in rating if "".join(s.split()) != ""]
book_info["豆瓣评分"] = "".join(new_rating)
```

爬取的电影信息包括：ID、基本信息、剧情简介、演职员表和评分信息；爬取的书籍信息包括：ID、基本信息、内容简介、作者简介和评分信息。将爬取到的信息存储在 CSV 文件中。

#### 反爬措施

豆瓣对于未登录用户的网页请求，规定同一 IP 地址在一定时间内最多请求100多次；但对于登录的用户，则没有这一限制。所以我们使用已登录账户的 Cookie 信息，模拟登录过的用户，从而绕过豆瓣的限制。

```Python
user_agent = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
}
cookies = # Cookie 篇幅较长，不便展示
```

### 检索

此环节将读取上节爬取的内容，以此分词和建立倒排索引，并进行布尔检索。最后，将对检索效果进行展示和评估。

#### 预处理

预处理包括将爬取的信息整合、分词、删除停用词等操作。

信息整合包括对收集数据进行处理，例如：电影《肖申克的救赎》类型为 剧情 / 犯罪，所以在信息整合部分将其作为两个词语添加进词项中。又比如收集的演职员表以下列方式存储：弗兰克·德拉邦特 导演/蒂姆·罗宾斯 饰 安迪·杜佛兰 Andy Dufresne/摩根·弗里曼 饰 艾利斯·波伊德·“瑞德”·瑞丁 Ellis Boyd 'Red' Redding... 对人名的划分和处理在这一部分进行。

##### 分词

分词部分采用了分词工具进行处理。通过对比两种分词工具：HanLP 和 THULAC，最终选用了 HanLP 作为项目的分词工具。

HanLP 的分词效果如下：

```Python
import hanlp

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
tok('晓美焰来到北京立方庭参观自然语义科技公司')
```

```commandline
['晓美焰', '来到', '北京立方庭', '参观', '自然语义科技公司']
```

THULAC 的分词效果如下：

```Python
import thulac

thu1 = thulac.thulac()
text = thu1.cut("晓美焰来到北京立方庭参观自然语义科技公司", text=True)
```

```commandline
晓_n 美_a 焰_n 来到_v 北京_ns 立方庭_n 参观_v 自然_a 语义_n 科技_n 公司_n
```

通过对比可知，虽然 THULAC 在默认模式下可以进行更为细致的划分，但是对特殊人名的处理却不理想；HanLP 可以在没有词典的情况下正确识别出人名，并且如果调用细分模型，效果与 THULAC 无异。

```Python
tok = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
tok('晓美焰来到北京立方庭参观自然语义科技公司')
```

```commandline
['晓美焰', '来到', '北京', '立方庭', '参观', '自然', '语义', '科技', '公司']
```

对于一段长文本，例如电影简介，HanLP 和 THULAC 的划分效果如下：

```Python
import hanlp
import thulac

text = """
       《哈利·波特与火焰杯》是“哈利·波特”系列的第四部。 哈利·波特在霍格沃茨魔法学校经过三年的学习和磨炼，逐渐成长为一个出色的巫师。新学年开始前，哈利和好朋友罗恩，赫敏一起去观看精彩的魁地奇世界杯赛，无意间发现了消失十三年的黑魔标记。
        哈利的心头笼上了一团浓重的阴云，但三个少年依然拥有他们自己的伊甸园。然而，少男少女的心思是那样难以捉摸，三人之间的美好友情竟是那样一波三折，忽晴忽雨……哈利渴望与美丽的秋·张共同走进一个美丽的故事，但这个朦朦胧胧的憧憬却遭受了小小的失意。他要做一个普普通通的四年级魔法学生，可不幸的是，哈利注定永远都不可能平平常常——即使拿魔法界的标准来衡量。
        黑魔的阴影始终挥之不去，种种暗藏杀机的神秘事件将哈利一步步推向了伏地魔的魔爪。哈利渴望在百年不遇的三强争霸赛中战胜自我，完成三个惊险艰巨的魔法项目，谁知整个竞赛竟是一个天大的黑魔法阴谋。……
        """

tok = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
thu = thulac.thulac(seg_only=True)
```

HanLP：

```commandline
['《', '哈利·波特', '与', '火焰杯', '》', '是', '“', '哈利·波特', '”', '系列', '的', '第四部', '。', '哈利·波特', '在', '霍格沃茨魔法学校', '经过', '三年', '的', '学习', '和', '磨炼', '，', '逐渐', '成长', '为', '一个', '出色', '的', '巫师', '。', '新', '学年', '开始', '前', '，', '哈利', '和', '好', '朋友', '罗恩', '，', '赫敏', '一起', '去', '观看', '精彩', '的', '魁地奇', '世界杯赛', '，', '无意间', '发现', '了', '消失', '十三年', '的', '黑魔', '标记', '。', '哈利', '的', '心头', '笼', '上', '了', '一', '团', '浓重', '的', '阴云', '，', '但', '三个', '少年', '依然', '拥有', '他们', '自己', '的', '伊甸园', '。', '然而', '，', '少男', '少女', '的', '心思', '是', '那样', '难以', '捉摸', '，', '三', '人', '之间', '的', '美好', '友情', '竟是', '那样', '一波三折', '，', '忽', '晴', '忽', '雨', '……', '哈利', '渴望', '与', '美丽', '的', '秋·张', '共同', '走进', '一个', '美丽', '的', '故事', '，', '但', '这个', '朦朦胧胧', '的', '憧憬', '却', '遭受', '了', '小小', '的', '失意', '。', '他', '要', '做', '一个', '普普通通', '的', '四年级', '魔法', '学生', '，', '可', '不幸', '的', '是', '，', '哈利', '注定', '永远', '都', '不可能', '平平常常', '——', '即使', '拿', '魔法界', '的', '标准', '来', '衡量', '。', '黑魔', '的', '阴影', '始终', '挥之不去', '，', '种种', '暗藏', '杀机', '的', '神秘', '事件', '将', '哈利', '一', '步', '步', '推向', '了', '伏地魔', '的', '魔爪', '。', '哈利', '渴望', '在', '百年', '不遇', '的', '三', '强', '争霸赛', '中', '战胜', '自我', '，', '完成', '三个', '惊险', '艰巨', '的', '魔法', '项目', '，', '谁知', '整个', '竞赛', '竟是', '一个', '天大', '的', '黑', '魔法', '阴谋', '。', '……']
```

THULAC：

```commandline
《 哈 利 · 波特 与 火焰杯 》 是 “ 哈 利 · 波特 ” 系列 的 第四 部 。 哈 利 · 波特 在 霍格沃茨 魔法 学校 经过 三 年 的 学习 和 磨炼 ， 逐渐 成长 为 一个 出色 的 巫师 。 新 学年 开始 前 ， 哈 利 和 好 朋友 罗 恩 ， 赫敏 一起 去 观看 精彩 的 魁地奇 世界杯赛 ， 无意间 发现 了 消失 十三 年 的 黑魔 标记 。 哈 利 的 心头笼 上 了 一 团 浓重 的 阴云 ， 但 三 个 少年 依然 拥有 他们 自己 的 伊甸园 。 然而 ， 少男少女 的 心思 是 那样 难以 捉摸 ， 三 人 之间 的 美好 友情 竟是 那样 一波三折 ， 忽 晴 忽 雨 … … 哈 利 渴望 与 美丽 的 秋 · 张 共同 走 进 一个 美丽 的 故事 ， 但 这个 朦朦胧胧 的 憧憬 却 遭受 了 小小 的 失意 。 他 要 做 一个 普普通通 的 四 年级 魔法 学生 ， 可 不 幸 的 是 ， 哈 利 注定 永远 都 不 可能 平平常常 —— 即使 拿 魔法界 的 标准 来 衡量 。 黑魔 的 阴影 始终 挥之不去 ， 种种 暗藏 杀机 的 神秘 事件 将 哈 利 一步步 推向 了 伏 地 魔 的 魔爪 。 哈 利 渴望 在 百年不遇 的 三 强 争霸赛 中 战胜 自我 ， 完成 三 个 惊险 艰巨 的 魔法 项目 ， 谁知 整个 竞赛 竟是 一个 天 大 的 黑 魔法 阴谋 。 … …
```

可以看出，两者各有优势：HanLP 在人名的处理方面显著优于 THULAC，而 THULAC 更加细粒度，“三年”进一步分成了”三“和”年“，虽然更加精细，但在添加了停用词的情况下，THULAC 的效果可能反而不如 HanLP。对于”忽晴忽雨“这样的非成语的四字词语，两者的效果都不是很好。

##### 停用词

首先制作停用词表，然后在遍历词项时，将出现的停用词删去，只将有效词语加入分词中。停用词表的设计参考了：哈工大停用词表、百度停用词表、nltk 中文停用词和英文停用词。

#### 倒排索引

遍历每个词项，统计出现该词项的文档集合，按照文档编号升序排列，生成倒排索引。

```Python
def get_inverted_index(tokens):
    inverted_index = []
    search_dict = {}
    all_tokens = set()

    for _, token in tokens.iterrows():
        _id = token["id"]
        token_set = set(token["tokens"].split())
        search_dict[_id] = token_set
        all_tokens.update(token_set)

    for _token in all_tokens:
        id_list = []
        for _id in search_dict.keys():
            if _token in search_dict[_id]:
                id_list.append(_id)
        id_list = sorted(id_list)
        id_list = [str(_id) for _id in id_list]
        inverted_index.append({"token": _token, "id_list": " ".join(id_list)})

    return inverted_index
```

针对跳表，在实验过程中我们设计了对每个词项建立跳表数据结构，内容如下。不过，在进一步研究后发现跳表除了增加代码负担之外并没有带来显著的性能提升。

```Python
import random

class SkipNode:
    def __init__(self, key, value=None):
        self.key = key
        self.value = value
        self.forward = []

class SkipList:
    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.head = SkipNode(float("-inf"))
        self.tail = SkipNode(float("inf"))
        self.head.forward = [self.tail] * (max_level + 1)
        self.p = p

    def random_level(self):
        level = 1
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, key):
        """Function to search"""

    def insert(self, key, value):
        update = [None] * (self.max_level + 1)
        current = self.head
        for level in range(self.max_level, -1, -1):
            while current.forward[level].key < key:
                current = current.forward[level]
            update[level] = current

        current = current.forward[0]

        if current.key == key:
            current.value = value
        else:
            level = self.random_level()
            new_node = SkipNode(key, value)
            if level > self.max_level:
                for i in range(self.max_level + 1, level + 1):
                    update.append(self.head)
                self.max_level = level
            for i in range(level + 1):
                new_node.forward.append(update[i].forward[i])
                update[i].forward[i] = new_node

    def delete(self, key):
        """Function to delete"""

    def display(self):
        """Function to display"""
```

#### 布尔检索

##### step1. 中缀布尔表达式转后缀布尔表达式

输入布尔表达式，首先对其进行处理，使他计算次序符合 `AND`, `OR`, `NOT` 的运算次序，例如 `A OR B AND C` 计算的次序为 `A OR (B AND C)`。

假设输入都是合法的，先将输入的中缀布尔表达式转化成后缀布尔表达式，例如 `A AND B` 转换成 `A B AND`。

具体步骤如下：

0. 设立了两个栈，分别存储 `操作符` 和 `字符`

1. 对于输入的语句，搜索 `AND` ，`OR`，`NOT`，`(`，`)` 所在的位置

   - 如果都不出现在语句中，说明语句处理完毕，跳转到 4

2. 判断搜寻到的操作符是否满足下列情况

   - `(`或者 `操作符栈`为空：
     - 直接将操作符加入对应栈中，并根据操作符类型选择是否将操作符前方临近的字符加入字符栈中

   - `AND`，`OR` 时直接添加

   - 如果 栈顶 的操作符优先级高于搜寻到的操作符优先级

     - 如果该操作符是 `)`
       - 将操作符栈的字符 `pop` 到 字符栈中，直到 `(` 

     - 否则，将操作符栈 `pop` 到字符栈中，直到不满足上述条件

3. 其他情况，则直接将操作符压入栈中，并根据操作符类型选择是否将操作符前方临近的字符加入字符栈中

4. 布尔查询语句字符处理完毕，将操作符栈中剩余的所有操作符压入字符栈中

```Python
PRIORITY_DICT = {"(": 0, ")": 4, "OR": 1, "AND": 2, "NOT": 3}

def infix_to_postfix(tokens):
    output = []
    operator_stack = []

    def is_operator(token):
        return token in {"AND", "OR", "NOT"}

    for token in tokens:
        if token == "(":
            operator_stack.append(token)
        elif token == ")":
            while operator_stack and operator_stack[-1] != "(":
                output.append(operator_stack.pop())
            operator_stack.pop()
        elif is_operator(token):
            while (
                operator_stack
                and operator_stack[-1] != "("
                and PRIORITY_DICT[operator_stack[-1]] > PRIORITY_DICT[token]
            ):
                output.append(operator_stack.pop())
            operator_stack.append(token)
        else:
            output.append(token)

    while operator_stack:
        output.append(operator_stack.pop())

    return output
```

##### step2. 计算后缀表达式

新建立一个栈 `stack`，存储布尔表达式中间过程和最终的计算结果。

```Python
def query_return(query, token_dict):
    stack = []
    tokens = tokenize(query)
    postfix = infix_to_postfix(tokens)
    search_dict = {}
    print(postfix)

    for token in postfix:
        if token not in OPT_DICT:
            stack.append(token_dict[token])
        elif token == "NOT":
            operand = stack.pop()
            stack.append(Not_opt(operand))
        elif token == "AND":
            operand2 = stack.pop()
            operand1 = stack.pop()
            stack.append(And_opt(operand1,operand2))
        elif token == "OR":
            operand2 = stack.pop()
            operand1 = stack.pop()
            stack.append(Or_opt(operand1,operand2))
    if stack:
        return stack[0]
    else:
        print("查找失败")
```

对于 `AND`, `OR` 和 `NOT` 运算，设计相关函数解决这个问题。对每一个词项及其对应的倒排索引，可以将其视为一个长度为 1200 的向量，对它们运算可以调用向量的相关函数。

```Python
import numpy as np

def And_opt(op1, op2):
    return np.logical_and(op1,op2)

def Or_opt(op1, op2):
    return np.logical_or(op1, op2)

def Not_opt(op):
    return np.logical_not(op)
```

#### 输出

如果通过布尔查询搜索到 ID 条目，可以通过其来调取爬取的相关数据，输出相关信息。下以书籍为例：

```Python
if (query_type == 1):
    inverted_lists = pd.read_csv("./invert/Book_invert.csv")
    data = pd.read_csv("./OriginData/BookData.csv")
    id_dict = {}  # 创建字典，将电影ID映射到0~1181，0~1181映射到ID
    for i in range(0, 1187):
        id_dict[int(data['ID'][i])] = i
        id_dict[i] = int(data['ID'][i])

    token_dict = {}  # 创建字典,存储id和对应的倒排表
    for idx, token in enumerate(inverted_lists['token']):
        try:
            inverted_list = [False] * 1187
            for i in inverted_lists['id_list'][idx].split(' '):
                inverted_list[id_dict[int(i)]] = True
            token_dict[token] = inverted_list
        except:
            continue
    ans_list = query_return(query, token_dict)
    ans = []
    num = 0
    for i, val in enumerate(ans_list):
        if val:
            num = num + 1
            ans.append(id_dict[i])
            print("第" + str(num) + "条检索结果：")
            print("书籍名称：" + str(data.loc[i]['书籍名称']))
            print("作者：" + str(data.loc[i]['作者']))
            print("出版社：" + str(data.loc[i]['出版社']))
            if data.loc[i]['原作名']!=None:
                print("原作名：" + str(data.loc[i]['原作名']))
            if data.loc[i]['译者'] != None:
                print("译者：" + str(data.loc[i]['译者']))
            print("出版年：" + str(data.loc[i]['出版年']))
            print("页数：" + str(data.loc[i]['页数']))
            print("定价：" + str(data.loc[i]['定价']))
            print("书籍简介：" + str(data.loc[i]['书籍简介']).replace(" ", ""))
            print("\n\n\n")
    print("共" + str(num) + "条检索结果")
    print("检索ID合集:")
    print(ans)
```

#### 检索效果

以下是对布尔检索效果的展示：

`query = "村上春树 AND 2001-2 AND 上海译文出版社"`

```commandline
['村上春树', '2001-2', '上海译文出版社', 'AND', 'AND']
第1条检索结果：
书籍名称：挪威的森林
作者：[日]村上春树
出版社：上海译文出版社
原作名：ノルウェイの森
译者：林少华
出版年：2001-2
页数：350
定价：18.80元
书籍简介：



这是一部动人心弦的、平缓舒雅的、略带感伤的恋爱小说。小说主人公渡边以第一人称展开他同两个女孩间的爱情纠葛。渡边的第一个恋人直子原是他高中要好同学木月的女友，后来木月自杀了。一年后渡边同直子不期而遇并开始交往。此时的直子已变得娴静腼腆，美丽晶莹的眸子里不时掠过一丝难以捕捉的阴翳。两人只是日复一日地在落叶飘零的东京街头漫无目标地或前或后或并肩行走不止。直子20岁生日的晚上两人发生了性关系，不料第二天直子便不知去向。几个月后直子来信说她住进一家远在深山里的精神疗养院。渡边前去探望时发现直子开始带有成熟女性的丰腴与娇美。晚间两人虽同处一室，但渡边约束了自己，分手前表示永远等待直子。返校不久，由于一次偶然相遇，渡边开始与低年级的绿子交往。绿子同内向的直子截然相反，“简直就像迎着春天的晨光蹦跳到世界上来的一头小鹿”。这期间，渡边内心十分苦闷彷徨。一方面念念不忘直子缠绵的病情与柔情，一方面又难以抗拒绿子大胆的表白和迷人的活力。不久传来直子自杀的噩耗，渡边失魂魄地四处徒步旅行。最后，在直子同房病友玲子的鼓励下，开始摸索此后的人生。






共1条检索结果
检索ID合集:
[1046265]
```

如果单纯检索 `村上春树` 、`2001-2` 或者 `上海译文出版社`，则分别有 33，2 和 94 条检索结果。

洪毓谦学号尾号为 10，豆瓣电影Top250的电影为《辛德勒的名单》。

`query = "史蒂文·斯皮尔伯格 AND (波兰 OR 纳粹)"`

```commandline
['史蒂文·斯皮尔伯格', '波兰', '纳粹', 'OR', 'AND']
第1条检索结果：
电影名称：辛德勒的名单 Schindler's List
别名：舒特拉的名单(港)/辛德勒名单
类型：剧情/历史/战争
导演：史蒂文·斯皮尔伯格
主演：连姆·尼森/本·金斯利/拉尔夫·费因斯/卡罗琳·古多尔/乔纳森·萨加尔/艾伯丝·戴维兹/马尔戈萨·格贝尔/马克·伊瓦涅/碧翠斯·马科拉/安德烈·瑟韦林/弗里德里希·冯·图恩/克齐斯茨托夫·拉夫特/诺伯特·魏塞尔/维斯瓦夫·科马萨
片长：195分钟
评分：9.5
电影简介：
　　1939年，波兰在纳粹德国的统治下，党卫军对犹太人进行了隔离统治。德国商人奥斯卡·辛德勒（连姆·尼森LiamNeeson饰）来到德军统治下的克拉科夫，开设了一间搪瓷厂，生产军需用品。凭着出众的社交能力和大量的金钱，辛德勒和德军建立了良好的关系，他的工厂雇用犹太人工作，大发战争财。

　　1943年，克拉科夫的犹太人遭到了惨绝人寰的大屠杀，辛德勒目睹这一切，受到了极大的震撼，他贿赂军官，让自己的工厂成为集中营的附属劳役营，在那些疯狂屠杀的日子里，他的工厂也成为了犹太人的避难所。

　　1944年，德国战败前夕，屠杀犹太人的行动越发疯狂，辛德勒向德军军官开出了1200人的名单，倾家荡产买下了这些犹太人的生命。在那些暗无天日的岁月里，拯救一个人，就是拯救全世界。





第2条检索结果：
电影名称：夺宝奇兵 Raiders of the Lost Ark
别名：法柜奇兵/夺宝奇兵：法柜奇兵/印地安纳・琼斯之夺宝奇兵/IndianaJonesandtheRaidersoftheLostArk
类型：动作/冒险
导演：史蒂文·斯皮尔伯格
主演：哈里森·福特/凯伦·阿兰/保罗·弗里曼/罗纳德·莱西/约翰·瑞斯-戴维斯/丹霍姆·艾略特/阿尔弗雷德·莫里纳/沃尔夫·卡赫勒/安东尼·希金斯/维克·塔布安/唐·费洛斯/威廉·胡特金斯
片长：115分钟
评分：7.9
电影简介：
　　二战期间，希特勒在世界各地召集考古学家寻找“失落的约柜”——圣经中引导希伯来人与上帝交流的圣物，希特勒欲借其来护佑纳粹的战争。

　　为了使希特勒的计划破灭，印第安纳琼斯博士（HarrisonFord饰）奔赴尼泊尔，一边奋力挖掘蛛丝马迹，一边还要与无孔不入的纳粹分子周旋。凭借着过人的智慧和异于常人的胆量及大无畏的勇气，琼斯终于在埃及找到了指示约柜位置的太阳手杖，继而发现了约柜。

　　就在大功告成之际，纳粹军却侵吞了胜利果实，还将琼斯等人留在了蛇穴里！但是无往不胜的琼斯总是有办法死里逃生，并狠狠地还以颜色！





共2条检索结果
检索ID合集:
[1295124, 1296717]
```

时数学号尾号为 52，豆瓣电影Top250电影是《绿皮书》。

`query = "维果·莫腾森 AND NOT 第二部"`

```commandline
['维果·莫腾森', '第二部', 'NOT', 'AND']
第1条检索结果：
电影名称：指环王1：护戒使者 The Lord of the Rings: The Fellowship of the Ring
别名：指环王1：魔戒再现/指环王I：护戒使者/魔戒首部曲：魔戒现身/魔戒1：护戒联盟
类型：剧情/动作/奇幻/冒险
导演：彼得·杰克逊
主演：伊利亚·伍德/西恩·奥斯汀/伊恩·麦克莱恩/维果·莫腾森/奥兰多·布鲁姆/多米尼克·莫纳汉/比利·博伊德/克里斯托弗·李/马尔顿·索克斯/梅根·爱德华兹/伊安·霍姆/凯特·布兰切特/阿兰·霍华德/马克·弗格森/肖恩·宾/萨拉·贝克/劳伦斯·马克奥雷/安迪·瑟金斯/彼得·麦肯齐/伊恩·穆内/克雷格·帕克/卡梅隆·罗德/约翰·瑞斯-戴维斯/丽芙·泰勒/大卫·韦瑟莱/雨果·维文/菲利普·格里夫/威廉·约翰逊/伊丽莎白·穆迪/布莱恩·瑟金特/杰德·布罗菲/诺曼·凯茨/兰德尔·威廉·库克/萨比恩·克洛森/西奥沙福瓦/本·弗兰舍姆/彼得·杰克逊/艾伦·李
片长：179分钟/208分钟(加长版)/228分钟(蓝光加长版)
评分：9.1
电影简介：
　　比尔博·巴金斯是100多岁的霍比特人，住在故乡夏尔，生性喜欢冒险，在年轻时的一次探险经历中，他从怪物咕噜手中得到了至尊魔戒，这枚戒指是黑暗魔君索伦打造的至尊魔戒，拥有奴役世界的邪恶力量，能够统领其他几枚力量之戒，在3000年前的人类联盟和半兽人大军的战役中，联盟取得了胜利，并得到了至尊魔戒，数千年的辗转后，魔戒落到咕噜手中，被比尔博碰巧得到。

　　因为和魔戒的朝夕相处，比尔博的心性也受到了影响，在他111岁的生日宴会上，他决定把一切都留给侄子佛罗多（伊利亚‧伍德饰)，继续冒险。

　　比尔博的好朋友灰袍巫师甘道夫（伊恩·麦克莱恩饰）知道至尊魔戒的秘密，同时，黑暗魔君索伦已经知道他的魔戒落在哈比族的手中。索伦正在重新建造要塞巴拉多，集结无数的半兽人，准备以大军夺取魔戒，并且征服全世界。

　　甘道夫说服佛罗多将魔戒护送到精灵王国瑞文希尔，佛罗多在好朋友山姆、皮平和梅利的陪同下，在跃马旅店得到了刚铎王子阿拉贡的帮助，历经艰难，终于到达了精灵王国。

　　然而，精灵族并不愿意保管这个邪恶的至尊魔戒，中土各国代表开会讨论，达成意见，准备将至尊魔戒送到末日山脉的烈焰中彻底销毁，佛罗多挺身而出接受了这个任务，这次，陪伴他的除了三个好朋友，还有甘道夫、阿拉贡、精灵莱戈拉斯（奥兰多‧布鲁姆饰）、人类博罗米尔、侏儒金利。

　　一路上，魔戒远征军除了要逃避索伦爪牙黑骑士和半兽人的追杀之外，更要抵抗至尊魔戒本身的邪恶诱惑，前途困难重重。





第2条检索结果：
电影名称：指环王3：王者无敌 The Lord of the Rings: The Return of the King
别名：魔戒三部曲：王者再临(台/港)/指环王III：王者无敌/魔戒3：王者归来/指环王3：国王归来/指环王3：皇上回宫(豆友译名)
类型：剧情/动作/奇幻/冒险
导演：彼得·杰克逊
主演：伊利亚·伍德/西恩·奥斯汀/维果·莫腾森/奥兰多·布鲁姆/伊恩·麦克莱恩/肖恩·宾/多米尼克·莫纳汉/丽芙·泰勒/约翰·贝西/凯特·布兰切特/比利·博伊德/萨德文·布罗菲/阿利斯泰尔·布朗宁/马尔顿·索克斯/伯纳德·希尔/伊安·霍姆/布鲁斯·霍普金斯/伊恩·休斯/劳伦斯·马克奥雷/诺埃尔·阿普利比/布雷特·麦肯齐/AlexandraAstin/SarahMcLeod/MaisyMcLeod-Riera/约翰·诺贝尔/PaulNorell/米兰达·奥图/布鲁斯·菲利普斯/沙恩·朗吉/约翰·瑞斯-戴维斯/托德·里彭/安迪·瑟金斯/HarrySinclair/乔尔·托贝克/卡尔·厄本/史蒂芬·乌瑞/雨果·维文/大卫·文翰/阿兰·霍华德/萨拉·贝克/RobertPollock/佩特·史密斯/杰德·布罗菲/菲利普·格里夫/布拉德·道里夫/克里斯托弗·李/布鲁斯·斯宾斯/吉诺·阿赛维多/JarlBenzon/JørnBenzon/RobertCatto/MichaelElsworth/彼得·杰克逊/SandroKopp/安德鲁·莱斯尼/约瑟夫·米卡-亨特/亨利·莫腾森/克雷格·帕克/克里斯蒂安·瑞沃斯/迈克尔·斯曼内科/霍华德·肖/约翰·斯蒂芬森/理查德·泰勒
片长：201分钟/254分钟(加长版)/263分钟(蓝光加长版)
评分：9.3
电影简介：
　　魔幻战争逐渐进入高潮阶段。霍比特人佛罗多（伊利亚·伍德ElijahWood饰）携带着魔戒，与伙伴山姆（西恩·奥斯汀SeanAstin饰）以及狡猾阴暗的咕噜等前往末日山，一路上艰难险阻不断，魔君索伦为阻止魔戒被销毁用尽全力阻挠。另一方面，白袍巫师甘道夫（伊恩·麦克莱恩IanMcKellen饰）率中土勇士们镇守刚铎首都——白城米那斯提里斯。魔兽大军压境，黑暗与光明的决战即将来临……

　　本片是“指环王三部曲”的终结篇，根据英国作家J.R.R.托尔金（J.R.R.Tolkien）同名魔幻巨著《指环王》（TheLordoftheRings）改编，并荣获2004年第76届奥斯卡最佳影片、最佳导演、最佳改编剧本、最佳剪辑、最佳艺术指导、最佳服装设计、最佳化妆、最佳视觉效果、最佳音效、最佳配乐和最佳歌曲等11项大奖。





共2条检索结果
检索ID合集:
[1291571, 1291552]
```

#### 索引压缩

对于倒排索引的压缩，采用了间距代替文档ID的压缩方式，可以使书籍倒排索引文件的存储空间从1.88M压缩到1.71M，电影倒排索引文件的存储空间从2.00M压缩到1.85M。

布尔检索层面，采用压缩后的倒排索引文件后的时间变化并不明显，检索效率可以视为无差异（从输入结束到输出结束的时间主要是由 `print` 函数占据）：

```commandline
query: 村上春树 AND 2001-2 AND 上海译文出版社
origin: 0.001003s
compressed: 0.001047s
```

## 第 2 阶段 使用豆瓣数据进行推荐

### 数据划分

实验按照训练集、验证集和测试集比例 8: 1: 1 构造，所用数据为 `movie_score.csv`，经预处理后得到 `user.dat`，`movie.dat` 和 `ratings.dat` 三个文件，然后建立 `MovieRatingDataset` 数据集。代码如下：

```Python
class MovieRatingDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        transform_args=None,
        pre_transform_args=None,
    ):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (process data).
        """
        super(MovieRatingDataset, self).__init__(root, transform, pre_transform)
        self.transform = transform
        self.pre_transform = pre_transform
        self.transform_args = transform_args
        self.pre_transform_args = pre_transform_args

    @property
    def raw_file_names(self):
        return "movie_score.zip"

    def processed_file_names(self):
        return ["movie_rating.pt"]

    def download(self):
        # Download to `self.raw_dir`.
        download_url(DATA_PATH, self.raw_dir)

    def _load(self):
        print(self.raw_dir)
        # extract_zip(self.raw_paths[0], self.raw_dir)
        with zipfile.ZipFile(self.raw_paths[0], "r") as zip_ref:
            zip_ref.extractall(self.raw_dir)
        unames = ["user_id"]
        users = pd.read_table(
            self.raw_dir + "/movie_score/users.dat",
            sep="::",
            header=None,
            names=unames,
            engine="python",
            encoding="latin-1",
        )

        rnames = ["user_id", "movie_id", "rating", "timestamp"]
        ratings = pd.read_table(
            self.raw_dir + "/movie_score/ratings.dat",
            sep="::",
            header=None,
            names=rnames,
            engine="python",
            encoding="latin-1",
        )

        mnames = ["movie_id"]
        movies = pd.read_table(
            self.raw_dir + "/movie_score/movies.dat",
            sep="::",
            header=None,
            names=mnames,
            engine="python",
            encoding="latin-1",
        )

        dat = pd.merge(pd.merge(ratings, users), movies)

        return users, ratings, movies, dat

    def process(self):
        print("run process")
        # load information from file
        users, ratings, movies, dat = self._load()

        users = users["user_id"]
        movies = movies["movie_id"]

        num_users = config_dict["num_users"]
        if num_users != -1:
            users = users[:num_users]

        user_ids = range(len(users))
        movie_ids = range(len(movies))

        user_to_id = dict(zip(users, user_ids))
        movie_to_id = dict(zip(movies, movie_ids))

        # get adjacency info
        self.num_user = users.shape[0]
        self.num_item = movies.shape[0]

        # initialize the adjacency matrix
        rat = torch.zeros(self.num_user, self.num_item)

        for index, row in ratings.iterrows():
            user, movie, rating = row[:3]
            if num_users != -1:
                if user not in user_to_id:
                    break
            # create ratings matrix where (i, j) entry represents the ratings of movie j given by user i.
            rat[user_to_id[user], movie_to_id[movie]] = rating

        # create Data object
        data = Data(
            edge_index=rat,
            raw_edge_index=rat.clone(),
            data=ratings,
            users=users,
            items=movies,
        )

        # apply any pre-transformation
        if self.pre_transform is not None:
            data = self.pre_transform(data, self.pre_transform_args)

        # apply any post_transformation
        # if self.transform is not None:
        #     # data = self.transform(data, self.transform_args)
        data = self.transform(data, [rating_threshold])

        # save the processed data into .pt file
        torch.save(data, osp.join(self.processed_dir, f"movie_rating.pt"))
        print("process finished")

    def len(self):
        # returns the number of examples in the graph
        return

    def get(self):
        # load a single graph
        data = torch.load(osp.join(self.processed_dir, "movie_rating.pt"))
        return data

    def train_val_test_split(self, val_frac=0.2, test_frac=0.1):
        # returns two mask matrices (M, N) that represents edges present in the train and validation set
        try:
            self.num_user, self.num_item
        except AttributeError:
            data = self.get()
            self.num_user = len(data["users"].unique())
            self.num_item = len(data["items"].unique())
        # get number of edges masked for training and validation
        num_train_replaced = round(
            (test_frac + val_frac) * self.num_user * self.num_item
        )
        num_val_show = round(val_frac * self.num_user * self.num_item)

        # edges masked during training
        indices_user = np.random.randint(0, self.num_user, num_train_replaced)
        indices_item = np.random.randint(0, self.num_item, num_train_replaced)

        # sample part of edges from training stage to be unmasked during
        # validation
        indices_val_user = np.random.choice(indices_user, num_val_show)
        indices_val_item = np.random.choice(indices_item, num_val_show)

        train_mask = torch.ones(self.num_user, self.num_item)
        train_mask[indices_user, indices_item] = 0

        val_mask = train_mask.clone()
        val_mask[indices_val_user, indices_val_item] = 1

        test_mask = torch.ones_like(train_mask)

        return train_mask, val_mask, test_mask
```

### LightGCN 模型

我们首先选择了 LightGCN 作为模型。LightGCN是一种将图卷积神经网络应用于推荐系统的算法，它是对神经图协同过滤（NGCF）算法的优化和改进。NGCF是基于图卷积网络（GCN）的协同过滤算法，但它在GCN的特征转换和非线性激活过程上存在一些问题。为了解决这些问题，LightGCN简化了标准GCN的设计，使其更适用于推荐任务。

具体来说，LightGCN的原理是通过在用户-物品交互图上进行图卷积操作来学习用户和物品的嵌入表示。它采用了一种简化的图卷积操作，即将用户和物品的嵌入向量相加而不进行特征转换和非线性激活。这种简化的操作可以减少模型的复杂性和计算量，同时保持了推荐任务中用户和物品之间的关系。

### 评分排序

我们训练了 10 个 ecpoch, 最后得到的结果如下列所示：

```
Training on 9 epoch completed.
 Average bpr_loss on train set is 0.00542 for the current epoch.
 Training top K precision = 0.4545000000000002, recall = 0.012467440579969673.
 Average bpr_loss on the validation set is 7e-06, and regularization loss is 0.0.
 Validation top K precision = 0.4545000000000001, recall = 0.012467440579969674.
 Average NDCG: 0.41525499999999993
```

NDCG 评分代码：

```Python
def compute_ndcg(group):
    true_ratings = group["true"].tolist()
    pred_ratings = group["pred"].tolist()
    return ndcg_score([true_ratings], [pred_ratings], k=50)
```

模型部分代码如下：

```Python
class LightGCNConv(MessagePassing):
    r"""The neighbor aggregation operator from the `"LightGCN: Simplifying and
    Powering Graph Convolution Network for Recommendation"
    <https://arxiv.org/abs/2002.02126#>`_ paper

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        num_users (int): Number of users for recommendation.
        num_items (int): Number of items to recommend.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_users: int,
        num_items: int,
        **kwargs,
    ):
        super(LightGCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_users = num_users
        self.num_items = num_items

        self.reset_parameters()

    def reset_parameters(self):
        pass  # There are no layer parameters to learn.

    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        """Performs neighborhood aggregation for user/item embeddings."""
        user_item = torch.zeros(self.num_users, self.num_items, device=x.device)
        user_item[edge_index[:, 0], edge_index[:, 1]] = 1
        user_neighbor_counts = torch.sum(user_item, axis=1)
        item_neightbor_counts = torch.sum(user_item, axis=0)

        item_neighbor_matrix = item_neightbor_counts.repeat(self.num_users, 1)
        # Compute weight for aggregation: 1 / (N_u)
        weights = user_item / (
            user_neighbor_counts.repeat(self.num_items, 1).T
            * torch.ones(item_neighbor_matrix.shape)
        )
        weights = torch.nan_to_num(weights, nan=0)
        out = torch.concat(
            (weights.T @ x[: self.num_users], weights @ x[self.num_users :]), 0
        )
        return out

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class LightGCN(nn.Module):
    def __init__(self, config: dict, device=None, **kwargs):
        super().__init__()

        self.num_users = config["n_users"]
        self.num_items = config["m_items"]
        self.embedding_size = config["embedding_size"]
        self.in_channels = self.embedding_size
        self.out_channels = self.embedding_size
        self.num_layers = config["num_layers"]

        # 0-th layer embedding.
        self.embedding_user_item = torch.nn.Embedding(
            num_embeddings=self.num_users + self.num_items,
            embedding_dim=self.embedding_size,
        )
        self.alpha = None

        # random normal init seems to be a better choice when lightGCN actually
        # don't use any non-linear activation function
        nn.init.normal_(self.embedding_user_item.weight, std=0.1)
        print("use NORMAL distribution initilizer")

        self.f = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.convs.append(
            LightGCNConv(
                self.embedding_size,
                self.embedding_size,
                num_users=self.num_users,
                num_items=self.num_items,
                **kwargs,
            )
        )

        for _ in range(1, self.num_layers):
            self.convs.append(
                LightGCNConv(
                    self.embedding_size,
                    self.embedding_size,
                    num_users=self.num_users,
                    num_items=self.num_items,
                    **kwargs,
                )
            )

        self.device = None
        if device is not None:
            self.convs.to(device)
            self.device = device

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, *args, **kwargs) -> Tensor:
        xs: List[Tensor] = []

        edge_index = torch.nonzero(edge_index)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, *args, **kwargs)
            if self.device is not None:
                x = x.to(self.device)
            xs.append(x)
        xs = torch.stack(xs)

        self.alpha = 1 / (1 + self.num_layers) * torch.ones(xs.shape)
        if self.device is not None:
            self.alpha = self.alpha.to(self.device)
            xs = xs.to(self.device)
        x = (xs * self.alpha).sum(dim=0)  # Sum along K layers.
        return x

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, num_layers={self.num_layers})"
        )
```

### LightGCN 结果分析

我们发现，虽然模型在训练集和验证集上的 bpr loss 较低，但整体的效果并不好。如 NDCG 值只有 0.415。

对数据进行分析后发现，pred 和 truth 的相差过大。

结合以上情况，我们不得不忍痛放弃了这个思路，转向了级联融合评分预测算法。

### 级联融合评分预测

类似于AdaBoost算法，即每次产生一个新模型，按照一定的参数加到旧模型上去，从而使训练集误差最小化。不同的是，这里每次生成新模型时并不对样本集采样，针对那些预测错的样本，而是每次都还是利用全样本集进行预测，但每次使用的模型都有区别，用来预测上一次的误差，并最后联合在一起预测。

假设已经有一个预测器，对于每个用户—物品对都给出预测值，那么可以在这个预测器的基础上设计下一个预测器来最小化损失函数。

最简单的融合模型是线性融合。 系数的选取一般采用如下方法：

- 假设数据集已经被分为了训练集A和测试集B，那么首先需要将训练集A按照相同的分割方法分为A1和A2，其中A2的生成方法和B的生成方法一致，且大小相似。
- 在A1上训练K个不同的预测器，在A2上作出预测。因为我们知道A2上的真实评分值，所以可以在A2上利用最小二乘法计算出线性融合系数
- 在A上训练K个不同的预测器，在B上作出预测，并且将这K个预测器在B上的预测结果按照已经得到的线性融合系数加权融合，以得到最终的预测结果。

我们选用了UserActivityCluster和ItemVoteCluster。

```Python
class UserActivityCluster(Cluster):
    def __init__(self, records):
        Cluster.__init__(self, records)
        activity = {}
        for r in records:
            if r.test:
                continue
            if r.user not in activity:
                activity[r.user] = 0
            activity[r.user] += 1
        # 按照用户活跃度进行分组
        k = 0
        for user, n in sorted(activity.items(), key=lambda x: x[-1], reverse=False):
            c = int((k * 5) / len(activity))
            self.group[user] = c
            k += 1

    def GetGroup(self, uid):
        if uid not in self.group:
            return -1
        else:
            return self.group[uid]


# 5. ItemVoteCluster
class ItemVoteCluster(Cluster):
    def __init__(self, records):
        Cluster.__init__(self, records)
        vote, cnt = {}, {}
        for r in records:
            if r.test:
                continue
            if r.item not in vote:
                vote[r.item] = 0
                cnt[r.item] = 0
            vote[r.item] += r.rate
            cnt[r.item] += 1
        # 按照物品平均评分进行分组
        for item, v in vote.items():
            c = v / (cnt[item] * 1.0)
            self.group[item] = int(c * 2)

    def GetGroup(self, iid):
        if iid not in self.group:
            return -1
        else:
            return self.group[iid]
```

### 结果分析

级联融合评分预测结果如下：

```commandline
Movie
100%|██████████| 1023/1023 [00:00<00:00, 5290.67it/s]
{'train_rmse': 1.7673984494557404, 'test_rmse': 1.7738258164647045}
NDCG = 0.8470422057711704
Book
100%|██████████| 4419/4419 [00:00<00:00, 16807.77it/s]
{'train_rmse': 2.0226016699673357, 'test_rmse': 2.033213996758488}
NDCG = 0.8337157536357758
```

Movie 和 Book 的 NDCG 分别为 0.847 和 0.834，说明模型对评分预测的排序较好。此外，我们还测试了训练集和测试集的RMSE。

## 实验总结

在实验过程中，我们实现了使用爬虫爬取数据，并实现了一个简单的布尔检索系统。我们熟悉了一些推荐模型，如 GraphRec、LightGCN等，学习了级联融合评分预测算法，收获了很多。