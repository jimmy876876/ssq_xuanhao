import requests
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import os
from ssq_analysis import analyze_red_balls, compare_with_previous
from number_predictor import NumberPredictor
import random
import math

def analyze_lottery_data(df):
    """分析所有期号的开奖数据（向量化实现以提升性能）"""

    # 计算上一期红球字符串 Series（向下移动一行）
    prev_red_series = df['红球'].shift(-1)

    # 使用 DataFrame.apply 结合已解析列表进行分析
    def _analyze_row(idx, red_str):
        analysis = analyze_red_balls(red_str)

        prev_str = prev_red_series.iloc[idx]
        if pd.notna(prev_str):
            repeat_count, diagonal_count = compare_with_previous(red_str, prev_str)
            analysis['重号个数'] = repeat_count
            analysis['斜号个数'] = diagonal_count
        else:
            analysis['重号个数'] = 0
            analysis['斜号个数'] = 0
        return analysis

    analysis_results = [
        _analyze_row(i, df.iloc[i]['红球']) for i in range(len(df))
    ]

    analysis_df = pd.DataFrame(analysis_results)

    # 补充基础字段列
    analysis_df['期号'] = df['期号'].values
    analysis_df['开奖日期'] = df['开奖日期'].values
    analysis_df['红球'] = df['红球'].values
    analysis_df['蓝球'] = df['蓝球'].values

    # 调整列顺序
    columns_order = ['期号', '开奖日期', '红球', '蓝球', '和值', '平均值', '尾数和值',
                    '奇号个数', '偶号个数', '奇偶偏差', '奇号连续', '偶号连续',
                    '大号个数', '小号个数', '大小偏差', '尾号组数', 'AC值',
                    '连号个数', '连号组数', '首尾差', '最大间距', '重号个数', '斜号个数']
    
    analysis_df = analysis_df[columns_order]
    return analysis_df

def fetch_ssq_data():
    # API接口地址
    url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
    
    # 设置请求头
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
        "Origin": "https://www.cwl.gov.cn",
        "Host": "www.cwl.gov.cn",
        "X-Requested-With": "XMLHttpRequest",
        "sec-ch-ua": "\"Not_A Brand\";v=\"8\", \"Chromium\";v=\"120\", \"Microsoft Edge\";v=\"120\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
    }
    
    all_lottery_data = []
    total_pages = (500 + 99) // 100  # 向上取整，确保获取500期数据
    
    try:
        for page in range(1, total_pages + 1):
            # 请求参数
            params = {
                "name": "ssq",
                "issueCount": "500",  # 获取最近500期数据
                "issueStart": "",
                "issueEnd": "",
                "dayStart": "",
                "dayEnd": "",
                "pageNo": str(page),
                "pageSize": "100",  # 每页100条数据
                "systemType": "PC"
            }
            
            # 添加重试机制
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"正在获取第 {page} 页数据...")
                    response = requests.get(url, headers=headers, params=params, timeout=10)
                    response.raise_for_status()
                    
                    # 解析JSON数据
                    data = response.json()
                    if data['state'] != 0:
                        print(f"获取数据失败: {data.get('message', '未知错误')}")
                        return None
                    
                    # 处理本页数据
                    for item in data['result']:
                        red_balls = item['red'].split(',')
                        blue_ball = item['blue']
                        
                        all_lottery_data.append({
                            '期号': item['code'],
                            '开奖日期': item['date'],
                            '红球': ' '.join(red_balls),
                            '蓝球': blue_ball
                        })
                    
                    # 如果已经获取了足够的数据，就退出
                    if len(all_lottery_data) >= 500:
                        all_lottery_data = all_lottery_data[:500]  # 只保留前500期
                        break
                    
                    # 在请求之间添加短暂延时，避免请求过于频繁
                    time.sleep(1)
                    break  # 成功获取数据，跳出重试循环
                    
                except requests.exceptions.RequestException as e:
                    print(f"第{attempt + 1}次请求失败: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # 失败后等待2秒再重试
                        continue
                    raise
            
            if len(all_lottery_data) >= 500:  # 如果数据够了就退出分页循环
                break
        
        # 将数据转换为DataFrame
        df = pd.DataFrame(all_lottery_data)
        
        # 保存为CSV文件
        df.to_csv('ssq_results.csv', index=False, encoding='utf-8-sig')
        print(f"成功获取 {len(df)} 期开奖数据，已保存到 ssq_results.csv")
        return df
                
    except Exception as e:
        print(f"处理数据时发生错误: {e}")
        return None

def load_data():
    """加载数据文件"""
    try:
        return pd.read_csv('ssq_results.csv')
    except FileNotFoundError:
        print("数据文件不存在，正在获取最新数据...")
        return fetch_ssq_data()
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        return None

def display_lottery_info(df, query=None):
    """显示开奖信息
    
    Args:
        df: DataFrame, 开奖数据
        query: str, 查询条件（期号、日期或None表示最新一期）
    """
    if df is None or df.empty:
        print("无法获取开奖数据，请先更新数据。")
        return
    
    result = None
    if query:
        # 尝试按期号查询
        result = df[df['期号'] == query]
        if result.empty:
            # 尝试按日期查询
            try:
                search_date = datetime.strptime(query, "%Y-%m-%d").strftime("%Y-%m-%d")
                result = df[df['开奖日期'].str.contains(search_date)]
            except ValueError:
                pass
    
    if result is None or result.empty:
        result = df.head(1)  # 获取最新一期
        if query:
            print(f"\n未找到与 '{query}' 匹配的开奖信息，显示最新一期数据：")
    
    for _, row in result.iterrows():
        red_numbers = [int(x) for x in row['红球'].split()]
        blue_number = int(row['蓝球'])
        
        print("\n=== 开奖信息 ===")
        print(f"期号：{row['期号']}")
        print(f"开奖日期：{row['开奖日期']}")
        print(f"中奖号码：红球 {format_numbers(red_numbers)} + 蓝球 {blue_number:02d}")
        
        print("\n号码分析：")
        print(f"和值：{row['和值']}")
        print(f"奇偶比：{row['奇号个数']}:{row['偶号个数']}")
        print(f"大小比：{row['大号个数']}:{row['小号个数']}")
        print(f"区间分布：", end='')
        zones = {'低区': 0, '中区': 0, '高区': 0}
        for num in red_numbers:
            if num <= 11:
                zones['低区'] += 1
            elif num <= 22:
                zones['中区'] += 1
            else:
                zones['高区'] += 1
        print(f"低区(1-11):{zones['低区']} 中区(12-22):{zones['中区']} 高区(23-33):{zones['高区']}")
        
        print("\n特征分析：")
        print(f"连号：{row['连号个数']}个 ({row['连号组数']}组)")
        print(f"重号：{row['重号个数']}个")
        print(f"斜号：{row['斜号个数']}个")
        print(f"AC值：{row['AC值']}")
        
        # 显示与上一期的对比
        if _ < len(df) - 1:
            prev_row = df.iloc[_ + 1]
            prev_red = set(int(x) for x in prev_row['红球'].split())
            curr_red = set(red_numbers)
            repeat_nums = curr_red & prev_red
            if repeat_nums:
                print(f"\n与上期重复号码：{format_numbers(repeat_nums)}")
            else:
                print("\n与上期无重复号码")

def display_trend_analysis(analysis_df, num_periods=10):
    """显示最近几期的走势分析"""
    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.width', None)  # 不限制宽度
    
    # 选择要显示的指标
    display_columns = [
        '期号', '开奖日期', '红球', '蓝球', '和值', '奇号个数', '偶号个数',
        '大号个数', '小号个数', '连号个数', '重号个数', '斜号个数'
    ]
    
    # 获取最近几期的数据
    recent_data = analysis_df[display_columns].head(num_periods)
    
    # 打印表头
    print("\n最近{}期开奖号码分析：".format(num_periods))
    print("-" * 120)
    
    # 打印每一期的数据
    for _, row in recent_data.iterrows():
        print("\n期号：{}  开奖日期：{}".format(row['期号'], row['开奖日期']))
        print("中奖号码：{} + {}".format(row['红球'], row['蓝球']))
        print("号码分析：")
        print("和值：{}  奇号：{}  偶号：{}  大号：{}  小号：{}  连号：{}  重号：{}  斜号：{}"
              .format(row['和值'], row['奇号个数'], row['偶号个数'],
                     row['大号个数'], row['小号个数'], row['连号个数'],
                     row['重号个数'], row['斜号个数']))
        print("-" * 120)

def display_detailed_analysis(row):
    """显示单期详细分析结果"""
    print("\n=== 详细号码分析 ===")
    print(f"期号：{row['期号']}")
    print(f"开奖日期：{row['开奖日期']}")
    print(f"中奖号码：{row['红球']} + {row['蓝球']}")
    print("\n基本指标：")
    print(f"和值：{row['和值']}    平均值：{row['平均值']}    尾数和值：{row['尾数和值']}")
    
    print("\n奇偶分析：")
    print(f"奇号个数：{row['奇号个数']}    偶号个数：{row['偶号个数']}    奇偶偏差：{row['奇偶偏差']}")
    print(f"奇号连续：{row['奇号连续']}    偶号连续：{row['偶号连续']}")
    
    print("\n大小分析：")
    print(f"大号个数：{row['大号个数']}    小号个数：{row['小号个数']}    大小偏差：{row['大小偏差']}")
    
    print("\n间距分析：")
    print(f"首尾差：{row['首尾差']}    最大间距：{row['最大间距']}    AC值：{row['AC值']}")
    
    print("\n连号分析：")
    print(f"连号个数：{row['连号个数']}    连号组数：{row['连号组数']}")
    
    print("\n重复性分析：")
    print(f"重号个数：{row['重号个数']}    斜号个数：{row['斜号个数']}    尾号组数：{row['尾号组数']}")

def check_data_freshness():
    """检查数据是否需要更新"""
    try:
        # 检查文件是否存在
        if not os.path.exists('ssq_results.csv'):
            return True
            
        # 读取现有数据
        df = pd.read_csv('ssq_results.csv')
        if df.empty:
            return True
            
        # 获取最新一期的日期
        latest_date_str = df.iloc[0]['开奖日期'].split('(')[0]  # 提取日期部分
        latest_date = datetime.strptime(latest_date_str, '%Y-%m-%d')
        
        # 如果最新数据超过1天，则更新
        if (datetime.now() - latest_date).days > 1:
            return True
            
        return False
    except Exception:
        return True

def initialize_data():
    """初始化或更新数据"""
    print("正在检查数据更新...")
    
    if check_data_freshness():
        print("需要更新数据...")
        df = fetch_ssq_data()
        if df is not None:
            print("正在分析数据...")
            analysis_df = analyze_lottery_data(df)
            return analysis_df
    else:
        print("数据是最新的，无需更新")
        try:
            df = pd.read_csv('ssq_results.csv')
            analysis_df = analyze_lottery_data(df)
            return analysis_df
        except Exception as e:
            print(f"读取数据时发生错误: {e}")
            return None
    
    return None

def display_number_recommendation(predictor):
    """显示号码推荐并返回推荐结果"""
    print("\n=== 双色球号码推荐 ===")
    
    # 分析历史数据规律
    hot_numbers, cold_numbers = predictor.analyze_hot_cold_numbers()
    avg_sum, std_sum = predictor.analyze_sum_value_distribution()
    target_odd = predictor.analyze_odd_even_pattern()
    target_big = predictor.analyze_big_small_pattern()
    
    print("\n历史数据分析：")
    print(f"热门号码（高频）：{sorted(hot_numbers)}")
    print(f"冷门号码（低频）：{sorted(cold_numbers)}")
    print(f"平均和值：{avg_sum:.2f} ± {std_sum:.2f}")
    print(f"最常见奇数个数：{target_odd}")
    print(f"最常见大号个数：{target_big}")
    
    # 生成推荐号码
    print("\n推荐号码组合：")
    combinations = predictor.generate_numbers(5)  # 生成5组号码
    
    # 存储分析结果
    recommendation_results = []
    
    for i, (red_numbers, blue_ball) in enumerate(combinations, 1):
        print(f"\n第{i}组：")
        red_str = ' '.join(f"{n:02d}" for n in red_numbers)
        print(f"红球：{red_str} + 蓝球：{blue_ball:02d}")
        
        # 显示号码分析
        analysis = predictor.get_number_analysis((red_numbers, blue_ball))
        print("号码分析：")
        print(f"和值：{analysis['和值']}")
        print(f"奇偶比：{analysis['奇号个数']}:{analysis['偶号个数']}")
        print(f"大小比：{analysis['大号个数']}:{analysis['小号个数']}")
        print(f"连号个数：{analysis['连号个数']}")
        print(f"AC值：{analysis['AC值']}")
        
        # 存储本组号码的信息
        result = {
            '组号': i,
            '红球': red_str,
            '蓝球': f"{blue_ball:02d}",
            '和值': analysis['和值'],
            '奇号个数': analysis['奇号个数'],
            '偶号个数': analysis['偶号个数'],
            '大号个数': analysis['大号个数'],
            '小号个数': analysis['小号个数'],
            '连号个数': analysis['连号个数'],
            'AC值': analysis['AC值']
        }
        recommendation_results.append(result)
    
    return recommendation_results

def export_recommendations(recommendations, filename=None):
    """导出号码推荐结果到文本文件"""
    if not recommendations:
        print("没有可导出的推荐号码")
        return
    
    if filename is None:
        # 使用当前时间生成默认文件名
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'ssq_recommendations_{current_time}.txt'
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== 双色球号码推荐 ===\n\n")
            for rec in recommendations:
                f.write(f"第{rec['组号']}组：{rec['红球']} + {rec['蓝球']}\n")
        print(f"\n推荐号码已成功导出到文件：{filename}")
    except Exception as e:
        print(f"导出文件时发生错误：{e}")

def generate_complex_betting_numbers(predictor, strategy='balanced'):
    """生成复式投注推荐号码
    
    Args:
        predictor: NumberPredictor实例
        strategy: 投注策略，可选值：
            - 'conservative': 保守策略 (7-8个红球，2-3个蓝球)
            - 'balanced': 平衡策略 (9-10个红球，3-4个蓝球)
            - 'aggressive': 激进策略 (11-12个红球，4-5个蓝球)
            - 'radical': 破釜沉舟策略 (13-15个红球，5-6个蓝球)
    
    Returns:
        tuple: (red_numbers, blue_numbers)，分别是推荐的红球号码列表和蓝球号码列表
    """
    # 根据策略确定选号数量
    strategy_config = {
        'conservative': {'red': (7, 8), 'blue': (2, 3)},
        'balanced': {'red': (9, 10), 'blue': (3, 4)},
        'aggressive': {'red': (11, 12), 'blue': (4, 5)},
        'radical': {'red': (13, 15), 'blue': (5, 6)}
    }
    
    if strategy not in strategy_config:
        strategy = 'balanced'
    
    config = strategy_config[strategy]
    
    # 分析历史数据
    hot_numbers, cold_numbers = predictor.analyze_hot_cold_numbers()
    avg_sum, std_sum = predictor.analyze_sum_value_distribution()
    target_odd = predictor.analyze_odd_even_pattern()
    target_big = predictor.analyze_big_small_pattern()
    
    # 确定红球选号数量
    red_count = random.randint(*config['red'])
    
    # 选择红球号码
    red_candidates = set()
    
    # 1. 先从热门号码中选择约60%的号码
    hot_count = int(red_count * 0.6)
    red_candidates.update(random.sample(hot_numbers, min(hot_count, len(hot_numbers))))
    
    # 2. 从冷门号码中选择约20%的号码
    cold_count = int(red_count * 0.2)
    remaining_cold = [n for n in cold_numbers if n not in red_candidates]
    if remaining_cold:
        red_candidates.update(random.sample(remaining_cold, min(cold_count, len(remaining_cold))))
    
    # 3. 剩余号码从1-33中随机选择
    remaining_count = red_count - len(red_candidates)
    if remaining_count > 0:
        all_numbers = set(range(1, 34))
        remaining_numbers = list(all_numbers - red_candidates)
        red_candidates.update(random.sample(remaining_numbers, remaining_count))
    
    # 确定蓝球选号数量
    blue_count = random.randint(*config['blue'])
    
    # 分析近期蓝球出现频率
    recent_blues = predictor.analyze_recent_blue_balls()
    
    # 选择蓝球号码
    blue_candidates = set()
    
    # 1. 先从最近常出现的蓝球中选择约70%
    blue_hot_count = int(blue_count * 0.7)
    blue_candidates.update(random.sample(recent_blues[:8], min(blue_hot_count, len(recent_blues[:8]))))
    
    # 2. 剩余从1-16中随机选择
    remaining_blue_count = blue_count - len(blue_candidates)
    if remaining_blue_count > 0:
        all_blue_numbers = set(range(1, 17))
        remaining_blue_numbers = list(all_blue_numbers - blue_candidates)
        blue_candidates.update(random.sample(remaining_blue_numbers, remaining_blue_count))
    
    return sorted(list(red_candidates)), sorted(list(blue_candidates))

def calculate_complex_betting_cost(red_count, blue_count):
    """计算复式投注注数和金额
    
    Args:
        red_count: 选择的红球数量
        blue_count: 选择的蓝球数量
    
    Returns:
        tuple: (注数, 金额)
    """
    # 计算红球组合数
    def combination(n, r):
        return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))
    
    # 计算注数
    red_combinations = combination(red_count, 6)  # 从red_count个红球中选6个的组合数
    total_combinations = red_combinations * blue_count  # 乘以蓝球数量
    
    # 每注2元
    total_cost = total_combinations * 2
    
    return total_combinations, total_cost

def display_complex_betting_recommendation(predictor):
    """显示复式投注推荐"""
    print("\n=== 双色球复式投注推荐 ===")
    
    strategies = {
        '1': ('conservative', '保守策略 (7-8红球 + 2-3蓝球)'),
        '2': ('balanced', '平衡策略 (9-10红球 + 3-4蓝球)'),
        '3': ('aggressive', '激进策略 (11-12红球 + 4-5蓝球)'),
        '4': ('radical', '破釜沉舟策略 (13-15红球 + 5-6蓝球)')
    }
    
    print("\n请选择投注策略：")
    for key, (_, desc) in strategies.items():
        print(f"{key}. {desc}")
    
    choice = input("\n请输入策略编号(1-4)，默认使用平衡策略：").strip()
    
    if choice not in strategies:
        choice = '2'  # 默认使用平衡策略
    
    strategy_name, strategy_desc = strategies[choice]
    
    print(f"\n已选择：{strategy_desc}")
    red_numbers, blue_numbers = predictor.generate_complex_numbers(strategy_name)
    
    print("\n=== 推荐号码组合 ===")
    print("红球：", ' '.join(f"{n:02d}" for n in red_numbers))
    print("蓝球：", ' '.join(f"{n:02d}" for n in blue_numbers))
    
    # 计算注数和金额
    combinations, cost = calculate_complex_betting_cost(len(red_numbers), len(blue_numbers))
    
    print(f"\n=== 投注信息 ===")
    print(f"红球个数：{len(red_numbers)}  蓝球个数：{len(blue_numbers)}")
    print(f"总注数：{combinations}注")
    print(f"投注金额：{cost}元")
    
    # 分析号码组合的中奖概率
    probability_analysis = predictor.analyze_winning_probability((red_numbers, blue_numbers[0]))  # 分析第一个蓝球组合
    
    print(f"\n=== 号码分析 ===")
    print(f"综合中奖概率评分：{probability_analysis['probability']:.2f}%")
    
    print("\n详细分析：")
    score_details = probability_analysis['score_details']
    print(f"1. 和值分析：{'✓' if score_details['和值合理性'] else '✗'}")
    print(f"2. 间隔分析：{'✓' if score_details['间隔合理性'] else '✗'}")
    print(f"3. 区域分布：{score_details['区域分布']:.2f}/1.00")
    print(f"4. 质数比例：{score_details['质数比例']:.2f} (偏差)")
    print(f"5. 连号分析：{score_details['连号合理性']}组连号")
    print(f"6. 遗漏值分析：{score_details['遗漏值分析']:.2f}/1.00")
    print(f"7. 蓝球分析：{'✓' if score_details['蓝球可能性'] else '✗'}")
    
    # 历史数据规律分析
    print("\n=== 历史规律分析 ===")
    print("1. 号码分区：")
    for zone, (start, end) in {'低区': (1, 11), '中区': (12, 22), '高区': (23, 33)}.items():
        count = len([n for n in red_numbers if start <= n <= end])
        print(f"   {zone}({start}-{end}): {count}个号码")
    
    # 计算和值
    sum_val = sum(red_numbers)
    print(f"\n2. 和值分析：")
    print(f"   当前和值：{sum_val}")
    print(f"   历史平均：{predictor.patterns['sum_range']['mean']:.2f}")
    print(f"   历史范围：{predictor.patterns['sum_range']['min']}-{predictor.patterns['sum_range']['max']}")
    
    # 奇偶分析
    odd_count = len([n for n in red_numbers if n % 2 == 1])
    even_count = len(red_numbers) - odd_count
    print(f"\n3. 奇偶分析：")
    print(f"   奇数：{odd_count}个")
    print(f"   偶数：{even_count}个")
    
    # 质数分析
    prime_count = len([n for n in red_numbers if predictor._is_prime(n)])
    print(f"\n4. 质数分析：")
    print(f"   质数：{prime_count}个")
    print(f"   合数：{len(red_numbers)-prime_count}个")
    
    if cost > 1000:
        print("\n提示：当前投注金额较大，建议考虑减少选号数量或选择更保守的策略")
    
    return {
        'strategy': strategy_desc,
        'red_numbers': red_numbers,
        'blue_numbers': blue_numbers,
        'combinations': combinations,
        'cost': cost,
        'probability': probability_analysis['probability'],
        'analysis': probability_analysis['score_details']
    }

def validate_numbers(numbers, min_count, max_count, min_val, max_val, number_type=""):
    """验证号码是否合法
    
    Args:
        numbers: list, 需要验证的号码列表
        min_count: int, 最小号码数量
        max_count: int, 最大号码数量
        min_val: int, 号码最小值
        max_val: int, 号码最大值
        number_type: str, 号码类型描述（用于错误提示）
    
    Returns:
        bool: 验证是否通过
    """
    try:
        if not (min_count <= len(numbers) <= max_count):
            print(f"{number_type}号码数量错误！需要{min_count}-{max_count}个号码。")
            return False
        if not all(min_val <= x <= max_val for x in numbers):
            print(f"{number_type}号码范围错误！需要{min_val}-{max_val}之间的号码。")
            return False
        return True
    except (ValueError, TypeError):
        print(f"{number_type}号码格式错误！")
        return False

def format_numbers(numbers):
    """格式化号码显示"""
    return ' '.join(f'{n:02d}' for n in sorted(numbers))

def check_winning_status(bet_numbers, winning_numbers):
    """检查投注号码的中奖情况"""
    bet_reds, bet_blue = bet_numbers
    win_reds, win_blue = winning_numbers
    
    # 计算红蓝球匹配数
    red_matches = len(set(bet_reds) & set(win_reds))
    blue_match = bet_blue == win_blue
    
    # 定义奖级规则
    prize_rules = [
        ((6, True), (1, "一等奖", "浮动")),
        ((6, False), (2, "二等奖", "浮动")),
        ((5, True), (3, "三等奖", 3000)),
        ((5, False), (4, "四等奖", 200)),
        ((4, True), (4, "四等奖", 200)),
        ((4, False), (5, "五等奖", 10)),
        ((3, True), (5, "五等奖", 10)),
        ((2, True), (6, "六等奖", 5)),
        ((1, True), (6, "六等奖", 5)),
        ((0, True), (6, "六等奖", 5))
    ]
    
    # 判断中奖等级
    for (req_red, req_blue), (level, name, amount) in prize_rules:
        if red_matches == req_red and (not req_blue or blue_match):
            return {
                'level': level,
                'name': name,
                'amount': amount,
                'red_matches': red_matches,
                'blue_match': blue_match
            }
    
    return {
        'level': 0,
        'name': "未中奖",
        'amount': 0,
        'red_matches': red_matches,
        'blue_match': blue_match
    }

def check_complex_betting_result(complex_numbers, winning_numbers):
    """检查复式投注的中奖情况"""
    red_numbers, blue_numbers = complex_numbers
    results = {
        'total_amount': 0,
        'prizes': {i: {'count': 0, 'amount': 0} for i in range(1, 7)},
        'details': []
    }
    
    # 检查每个组合
    for blue in blue_numbers:
        for reds in combinations(red_numbers, 6):
            result = check_winning_status((reds, blue), winning_numbers)
            if result['level'] > 0:
                level = result['level']
                results['prizes'][level]['count'] += 1
                if isinstance(result['amount'], (int, float)):
                    results['prizes'][level]['amount'] += result['amount']
                results['details'].append({
                    'red_numbers': sorted(reds),
                    'blue_number': blue,
                    'prize': result
                })
    
    # 计算固定奖金总额
    results['total_amount'] = sum(prize['amount'] 
                                for level, prize in results['prizes'].items() 
                                if level >= 3)
    
    return results

def display_winning_check_menu(predictor):
    """显示中奖查询菜单"""
    if predictor.df is None or predictor.df.empty:
        print("无法获取最新开奖数据，请先更新数据。")
        return
    
    # 获取最新开奖信息
    latest_draw = predictor.df.iloc[0]
    win_red_numbers = [int(x) for x in latest_draw['红球'].split()]
    win_blue_number = int(latest_draw['蓝球'])
    
    # 显示菜单
    print("\n=== 双色球中奖查询 ===")
    print(f"最新开奖期号：{latest_draw['期号']}")
    print(f"开奖日期：{latest_draw['开奖日期']}")
    print(f"开奖号码：红球 {format_numbers(win_red_numbers)} + 蓝球 {win_blue_number:02d}")
    print("\n1. 单式投注查询")
    print("2. 复式投注查询")
    print("3. 返回主菜单")
    
    choice = input("\n请选择操作 (1-3): ").strip()
    
    try:
        if choice == "1":
            # 单式投注查询
            print("\n请输入您的投注号码：")
            red_input = [int(x) for x in input("红球号码（用空格分隔6个号码）: ").strip().split()]
            if not validate_numbers(red_input, 6, 6, 1, 33, "红球"):
                return
            
            blue_number = int(input("蓝球号码（1-16之间的数字）: ").strip())
            if not validate_numbers([blue_number], 1, 1, 1, 16, "蓝球"):
                return
            
            # 检查中奖
            result = check_winning_status(
                (red_input, blue_number),
                (win_red_numbers, win_blue_number)
            )
            
            # 显示结果
            print("\n=== 中奖结果 ===")
            print(f"您的号码：红球 {format_numbers(red_input)} + 蓝球 {blue_number:02d}")
            print(f"开奖期号：{latest_draw['期号']}")
            print(f"开奖日期：{latest_draw['开奖日期']}")
            print(f"开奖号码：红球 {format_numbers(win_red_numbers)} + 蓝球 {win_blue_number:02d}")
            print(f"\n中奖等级：{result['name']}")
            print(f"中红球数：{result['red_matches']}个")
            print(f"中蓝球：{'是' if result['blue_match'] else '否'}")
            if result['amount'] == '浮动':
                print(f"奖金：{result['amount']}（具体金额以实际派奖为准）")
            else:
                print(f"奖金：{result['amount']}元")
                
        elif choice == "2":
            # 复式投注查询
            print("\n请输入您的复式投注号码：")
            red_input = [int(x) for x in input("红球号码（用空格分隔7-20个号码）: ").strip().split()]
            if not validate_numbers(red_input, 7, 20, 1, 33, "红球"):
                return
            
            blue_input = [int(x) for x in input("蓝球号码（用空格分隔1-16个号码）: ").strip().split()]
            if not validate_numbers(blue_input, 1, 16, 1, 16, "蓝球"):
                return
            
            # 检查中奖
            results = check_complex_betting_result(
                (red_input, blue_input),
                (win_red_numbers, win_blue_number)
            )
            
            # 显示结果
            print("\n=== 复式投注中奖结果 ===")
            print(f"您的号码：")
            print(f"红球：{format_numbers(red_input)}")
            print(f"蓝球：{format_numbers(blue_input)}")
            print(f"\n开奖期号：{latest_draw['期号']}")
            print(f"开奖日期：{latest_draw['开奖日期']}")
            print(f"开奖号码：红球 {format_numbers(win_red_numbers)} + 蓝球 {win_blue_number:02d}")
            
            # 显示中奖明细
            total_count = sum(prize['count'] for prize in results['prizes'].values())
            if total_count == 0:
                print("\n很遗憾，本次未中奖")
            else:
                print("\n中奖明细：")
                for level in range(1, 7):
                    count = results['prizes'][level]['count']
                    if count > 0:
                        amount = results['prizes'][level]['amount']
                        if level <= 2:
                            print(f"{level}等奖：{count}注（浮动奖金）")
                        else:
                            print(f"{level}等奖：{count}注，奖金{amount}元")
                
                if results['total_amount'] > 0:
                    print(f"\n固定奖金总额：{results['total_amount']}元")
                    if any(results['prizes'][i]['count'] > 0 for i in [1, 2]):
                        print("（不含一、二等奖浮动奖金）")
                
                if results['details']:
                    print("\n中奖组合详情：")
                    for detail in results['details']:
                        print(f"红球：{format_numbers(detail['red_numbers'])} + "
                              f"蓝球：{detail['blue_number']:02d} => {detail['prize']['name']}")
                              
    except ValueError:
        print("输入格式错误！请输入正确的数字。")
        return

def analyze_and_display_data(df, periods=10, export=False):
    """综合数据分析与显示
    
    Args:
        df: DataFrame, 开奖数据
        periods: int, 分析期数
        export: bool, 是否导出数据
    """
    if df is None or df.empty:
        print("无法获取数据，请先更新数据。")
        return
    
    # 1. 基础统计信息
    print("\n=== 基础统计信息 ===")
    print(f"总期数：{len(df)} 期")
    print(f"最早期号：{df.iloc[-1]['期号']} ({df.iloc[-1]['开奖日期']})")
    print(f"最新期号：{df.iloc[0]['期号']} ({df.iloc[0]['开奖日期']})")
    
    # 2. 号码统计
    print("\n=== 号码统计 ===")
    # 红球频率统计
    red_counts = {}
    for i in range(len(df)):
        numbers = [int(x) for x in df.iloc[i]['红球'].split()]
        for num in numbers:
            red_counts[num] = red_counts.get(num, 0) + 1
    
    # 蓝球频率统计
    blue_counts = {}
    for i in range(len(df)):
        num = int(df.iloc[i]['蓝球'])
        blue_counts[num] = blue_counts.get(num, 0) + 1
    
    # 显示出现次数最多和最少的号码
    print("\n红球出现频率：")
    sorted_red = sorted(red_counts.items(), key=lambda x: (-x[1], x[0]))
    print("高频号码（出现次数）：", end='')
    print(" ".join(f"{num:02d}({count})" for num, count in sorted_red[:5]))
    print("低频号码（出现次数）：", end='')
    print(" ".join(f"{num:02d}({count})" for num, count in sorted_red[-5:]))
    
    print("\n蓝球出现频率：")
    sorted_blue = sorted(blue_counts.items(), key=lambda x: (-x[1], x[0]))
    print("高频号码（出现次数）：", end='')
    print(" ".join(f"{num:02d}({count})" for num, count in sorted_blue[:5]))
    print("低频号码（出现次数）：", end='')
    print(" ".join(f"{num:02d}({count})" for num, count in sorted_blue[-5:]))
    
    # 3. 数值特征统计
    print("\n=== 数值特征统计 ===")
    print(f"和值范围：{df['和值'].min()}-{df['和值'].max()}, 平均：{df['和值'].mean():.2f}")
    print(f"AC值范围：{df['AC值'].min()}-{df['AC值'].max()}, 平均：{df['AC值'].mean():.2f}")
    print(f"连号：平均{df['连号个数'].mean():.2f}个，{df['连号组数'].mean():.2f}组")
    print(f"重号：平均{df['重号个数'].mean():.2f}个")
    print(f"斜号：平均{df['斜号个数'].mean():.2f}个")
    
    # 4. 最近走势分析
    recent_data = df.head(periods)
    print(f"\n=== 最近{periods}期走势分析 ===")
    
    # 计算遗漏值
    all_red_numbers = set(range(1, 34))
    all_blue_numbers = set(range(1, 17))
    missing_reds = {num: 0 for num in all_red_numbers}
    missing_blues = {num: 0 for num in all_blue_numbers}
    
    for _, row in recent_data.iterrows():
        red_nums = set(int(x) for x in row['红球'].split())
        blue_num = int(row['蓝球'])
        
        # 更新遗漏值
        for num in all_red_numbers - red_nums:
            missing_reds[num] += 1
        for num in all_blue_numbers - {blue_num}:
            missing_blues[num] += 1
    
    # 显示当前遗漏值较大的号码
    print("\n红球遗漏值统计（遗漏值>=5的号码）：")
    high_missing_reds = {k: v for k, v in missing_reds.items() if v >= 5}
    if high_missing_reds:
        for num, missing in sorted(high_missing_reds.items(), key=lambda x: (-x[1], x[0])):
            print(f"{num:02d}:{missing}期", end=' ')
        print()
    else:
        print("无遗漏值>=5的红球")
    
    print("\n蓝球遗漏值统计（遗漏值>=3的号码）：")
    high_missing_blues = {k: v for k, v in missing_blues.items() if v >= 3}
    if high_missing_blues:
        for num, missing in sorted(high_missing_blues.items(), key=lambda x: (-x[1], x[0])):
            print(f"{num:02d}:{missing}期", end=' ')
        print()
    else:
        print("无遗漏值>=3的蓝球")
    
    # 5. 近期特征分析
    print(f"\n近{periods}期特征分析：")
    for i, row in recent_data.iterrows():
        red_numbers = [int(x) for x in row['红球'].split()]
        blue_number = int(row['蓝球'])
        
        # 计算区间分布
        zones = {'低区': 0, '中区': 0, '高区': 0}
        for num in red_numbers:
            if num <= 11:
                zones['低区'] += 1
            elif num <= 22:
                zones['中区'] += 1
            else:
                zones['高区'] += 1
        
        print(f"\n{row['期号']} ({row['开奖日期']}):")
        print(f"号码：红球 {format_numbers(red_numbers)} + 蓝球 {blue_number:02d}")
        print(f"特征：和值{row['和值']} AC值{row['AC值']} " +
              f"连号{row['连号个数']}({row['连号组数']}组) " +
              f"重号{row['重号个数']} 斜号{row['斜号个数']}")
        print(f"分布：低区{zones['低区']} 中区{zones['中区']} 高区{zones['高区']} " +
              f"奇数{row['奇号个数']} 偶数{row['偶号个数']} " +
              f"大号{row['大号个数']} 小号{row['小号个数']}")
    
    # 6. 导出分析数据
    if export:
        try:
            # 导出完整数据
            output_file = 'ssq_analysis.xlsx'
            df.to_excel(output_file, index=False)
            print(f"\n完整分析数据已导出到：{output_file}")
            
            # 导出近期分析数据
            recent_file = f'ssq_recent_{periods}periods.xlsx'
            recent_data.to_excel(recent_file, index=False)
            print(f"近期分析数据已导出到：{recent_file}")
        except Exception as e:
            print(f"导出数据时发生错误：{e}")

def display_comprehensive_recommendations(predictor):
    """综合选号推荐功能
    
    Args:
        predictor: NumberPredictor实例
    """
    if predictor is None:
        print("请先更新数据")
        return None
    
    def _print_menu():
        print("\n=== 双色球选号推荐系统 ===")
        print("1. 单式投注推荐")
        print("2. 复式投注推荐")
        print("3. 查看历史推荐")
        print("4. 导出推荐结果")
        print("5. 号码分析评测")
        print("6. 中奖查询")
        print("7. 退出")

    _print_menu()
    recommendations = None
    while True:
        choice = input("\n请选择操作 (1-7): ").strip()
        
        if choice == "1":
            print("\n=== 单式投注推荐 ===")
            recommendations = display_number_recommendation(predictor)
            
            save = input("\n是否保存本次推荐结果？(y/n): ").lower().strip()
            if save == 'y':
                export_recommendations(recommendations)
                
            # 推荐结束后重新显示菜单
            _print_menu()
                
        elif choice == "2":
            print("\n=== 复式投注推荐 ===")
            recommendation = display_complex_betting_recommendation(predictor)
            recommendations = recommendation  # 保存最近一次推荐结果
            
            save = input("\n是否保存本次推荐结果？(y/n): ").lower().strip()
            if save == 'y':
                current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f'complex_betting_{current_time}.txt'
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("=== 双色球复式投注推荐 ===\n\n")
                        f.write(f"策略：{recommendation['strategy']}\n")
                        f.write(f"红球：{' '.join(f'{n:02d}' for n in recommendation['red_numbers'])}\n")
                        f.write(f"蓝球：{' '.join(f'{n:02d}' for n in recommendation['blue_numbers'])}\n")
                        f.write(f"\n总注数：{recommendation['combinations']}注\n")
                        f.write(f"投注金额：{recommendation['cost']}元\n")
                    print(f"\n推荐结果已保存到文件：{filename}")
                except Exception as e:
                    print(f"保存文件时发生错误：{e}")
                    
            # 复式推荐结束后重新显示菜单
            _print_menu()
                
        elif choice == "3":
            if recommendations is None:
                print("暂无历史推荐记录，请先使用选号推荐功能")
                continue
                
            print("\n=== 最近一次推荐结果 ===")
            if isinstance(recommendations, list):  # 单式推荐结果
                for i, rec in enumerate(recommendations, 1):
                    print(f"\n第{rec['组号']}组：")
                    print(f"红球：{rec['红球']}")
                    print(f"蓝球：{rec['蓝球']}")
                    print(f"和值：{rec['和值']}")
                    print(f"奇偶比：{rec['奇号个数']}:{rec['偶号个数']}")
                    print(f"大小比：{rec['大号个数']}:{rec['小号个数']}")
            else:  # 复式推荐结果
                print(f"推荐策略：{recommendations['strategy']}")
                print(f"红球：{' '.join(f'{n:02d}' for n in recommendations['red_numbers'])}")
                print(f"蓝球：{' '.join(f'{n:02d}' for n in recommendations['blue_numbers'])}")
                print(f"总注数：{recommendations['combinations']}注")
                print(f"投注金额：{recommendations['cost']}元")
                print(f"综合中奖概率评分：{recommendations['probability']:.2f}%")
                
            _print_menu()
                
        elif choice == "4":
            if recommendations is None:
                print("暂无推荐结果可导出，请先使用选号推荐功能")
                continue
                
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            if isinstance(recommendations, list):
                filename = f'ssq_recommendations_{current_time}.txt'
                export_recommendations(recommendations, filename)
            else:
                filename = f'ssq_complex_recommendations_{current_time}.txt'
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write("=== 双色球复式投注推荐 ===\n\n")
                        f.write(f"推荐策略：{recommendations['strategy']}\n\n")
                        f.write(f"红球：{' '.join(f'{n:02d}' for n in recommendations['red_numbers'])}\n")
                        f.write(f"蓝球：{' '.join(f'{n:02d}' for n in recommendations['blue_numbers'])}\n\n")
                        f.write(f"总注数：{recommendations['combinations']}注\n")
                        f.write(f"投注金额：{recommendations['cost']}元\n")
                        f.write(f"综合中奖概率评分：{recommendations['probability']:.2f}%\n\n")
                        f.write("=== 分析详情 ===\n")
                        for key, value in recommendations['analysis'].items():
                            f.write(f"{key}: {value}\n")
                    print(f"\n推荐结果已导出到文件：{filename}")
                except Exception as e:
                    print(f"导出文件时发生错误：{e}")
                    
            _print_menu()
                
        elif choice == "5":
            if predictor is None:
                print("请先更新数据")
                continue
            analyze_user_numbers(predictor)
            
            _print_menu()
                
        elif choice == "6":
            if predictor is None:
                print("请先更新数据")
                continue
            display_winning_check_menu(predictor)
            
            _print_menu()
                
        elif choice == "7":
            print("\n感谢使用，再见！")
            break

# ======================= 用户号码分析评测 =======================

def analyze_user_numbers(predictor):
    """让用户输入号码并进行详细分析评测"""
    print("\n=== 号码分析评测 ===")

    # 输入红球号码
    try:
        red_input = [int(x) for x in input("请输入红球号码（空格分隔6个1-33的数字）: ").strip().split()]
        if not validate_numbers(red_input, 6, 6, 1, 33, "红球"):
            return
        # 输入蓝球号码
        blue_input = int(input("请输入蓝球号码（1-16）: ").strip())
        if not validate_numbers([blue_input], 1, 1, 1, 16, "蓝球"):
            return
    except ValueError:
        print("输入格式错误！")
        return

    # 基础指标分析
    analysis = predictor.get_number_analysis((red_input, blue_input))

    # 概率评分
    prob_analysis = predictor.analyze_winning_probability((red_input, blue_input))

    print("\n=== 分析结果 ===")
    print(f"您输入的号码：红球 {' '.join(f'{n:02d}' for n in sorted(red_input))} + 蓝球 {blue_input:02d}")
    print("\n基本指标：")
    print(f"和值：{analysis['和值']}")
    print(f"奇偶比：{analysis['奇号个数']}:{analysis['偶号个数']}")
    print(f"大小比：{analysis['大号个数']}:{analysis['小号个数']}")
    print(f"连号个数：{analysis['连号个数']}")
    print(f"AC值：{analysis['AC值']}")

    # 评分详情
    print("\n综合中奖概率评分：{:.2f}%".format(prob_analysis['probability']))
    details = prob_analysis['score_details']
    print("\n详细评分：")
    print(f"1. 和值合理性：{'✓' if details['和值合理性'] else '✗'}")
    print(f"2. 间隔合理性：{'✓' if details['间隔合理性'] else '✗'}")
    print(f"3. 区域分布得分：{details['区域分布']:.2f}/1.00")
    print(f"4. 质数比例偏差：{details['质数比例']:.2f}")
    print(f"5. 连号个数：{details['连号合理性']} 组")
    print(f"6. 遗漏值分析得分：{details['遗漏值分析']:.2f}/1.00")
    print(f"7. 蓝球热门：{'✓' if details['蓝球可能性'] else '✗'}")