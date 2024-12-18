# 导入库
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def replace_industry_code(group):
    '''
    将2010年和2011年的行业代码替换为2012年的行业代码
    '''
    code_2012 = group.loc[group['EndDate'] == '2012-12-31', 'IndustryCode']
    if not code_2012.empty:
        group.loc[group['EndDate'].isin(['2010-12-31', '2011-12-31']), 'IndustryCode'] = code_2012.values[0]
    return group


def update_indus_code(df):
    '''
    处理国际专利分类与国民经济行业分类关系表
    '''
    current_letter = None
    current_code = None
    
    for i in range(len(df) - 1):  
        indus_code = str(df.loc[i, 'indus_code'])  
        patent_code = df.loc[i, 'patent_code']
        
        # 如果indus_code是大写字母，更新current_letter
        if indus_code.isupper():
            current_letter = indus_code
        
        # 如果indus_code是两个数字，将current_code更新为current_letter加这两个数字
        elif indus_code.isdigit() and len(indus_code) == 2:
            current_code = current_letter + indus_code
            print(f"Current Code: {current_code}")
        
        # 如果indus_code为NaN, 用current_code更新当前行
        elif indus_code == 'nan' or not indus_code:  
            if current_code: 
                df.loc[i, 'indus_code'] = current_code
                print(f"Updated indus_code at row {i}: {df.loc[i, 'indus_code']}")
    
    return df


# 处理专利细分分类表
patent_invention = pd.read_excel('上市公司实用新型获得专利细分分类号.xlsx',converters={'stock_code': str})
def add_industry_code(patent_invention, indus):
    '''
    将行业代码添加到专利发明表中
    '''    
    result = pd.merge(patent_invention, indus[['Symbol', 'year', 'IndustryCode']],
                      left_on=['stock_code', 'Year'], 
                      right_on=['Symbol', 'year'], 
                      how='left')  
    
    return result


def preprocess_relation(relation_adjust):
    """
    将relation1转换为嵌套字典结构，格式为 {indus_code: set_of_patent_codes}
    """
    relation_dict = {}
    for indus_code, patent_code in relation1[['indus_code', 'patent_code']].values:
        if indus_code not in relation_dict:
            relation_dict[indus_code] = set()
        relation_dict[indus_code].add(patent_code)
    return relation_dict


def process_row(args):
    """
    处理每一行的数据，计算主业专利数量（mbp）
    """
    i, row, patent_invention1, num_cols, relation_dict = args
    mbp = 0
    industry_code = row.IndustryCode

    for col in patent_invention1.columns:
        if col not in ['stock_code', 'Year', 'IndustryCode', 'mbp', 'Ftyp', 'Athrtm', 'year', 'Symbol'] + num_cols:
            patents = getattr(row, col)
            if pd.notna(patents) and isinstance(patents, str):  
                patent_groups = patents.strip('{}').split('},{')
                patent_groups = ['{' + group + '}' for group in patent_groups] 

                for patent_group in patent_groups:
                    # 拆分专利号并清除括号和I, 删除"/"及其后面的字符
                    patent_list = patent_group.strip('{}').split(';')
                    patent_list = [patent.split('/')[0] for patent in patent_list]

                    # 3. 判断每个列表是否包含主业专利
                    for patent in patent_list:
                        if industry_code in relation_dict and patent in relation_dict[industry_code]:
                            mbp += 1
                            break  # 每个列表最多只能有一个主业专利，找到一个就跳出
    return mbp


def count_mbp(patent_invention1, relation1):
    # 初始化mbp列
    patent_invention1['mbp'] = 0

    # 计算patent_count列：按行计算以num结尾的列的值相加
    num_cols = [col for col in patent_invention1.columns if col.endswith('num')]
    patent_invention1['patent_count'] = patent_invention1[num_cols].sum(axis=1)

    # 优化relation1为字典
    relation_dict = preprocess_relation(relation1)

    # 使用并行计算处理每一行
    args_list = [
        (i, row, patent_invention1, num_cols, relation_dict) 
        for i, row in enumerate(patent_invention1.itertuples(index=False))
    ]

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(process_row, args_list),
            total=len(patent_invention1),
            desc="Processing rows"
        ))

    # 更新mbp列
    patent_invention1['mbp'] = results
    return patent_invention1


if __name__ == '__main__':

    # 读取每年行业代码（CSMAR获取国民经济行业分类）
    indus = pd.read_csv('国民经济行业分类_csmar.csv', converters={'Symbol': str}) 
    indus['EndDate'] = pd.to_datetime(indus['EndDate'])

    # 添加年份
    indus = indus.groupby('Symbol').apply(replace_industry_code)
    indus = indus.reset_index(drop=True)
    indus['year'] = indus['EndDate'].dt.year

    # 读取关系表并调整(百度搜索可得)
    relation = pd.read_csv('关系表.csv', encoding='gbk')
    relation_adjust = update_indus_code(relation)

    # 获取调整后的专利发明表
    patent_invention_adjust = add_industry_code(patent_invention, indus)

    # 获得主页专利数量
    MBP_df = count_mbp(patent_invention_adjust, relation_adjust)


