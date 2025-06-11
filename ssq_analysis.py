import numpy as np
from itertools import combinations

def analyze_red_balls(red_balls_str):
    """分析红球号码的各项指标"""
    # 将红球字符串转换为数字列表
    red_balls = [int(x) for x in red_balls_str.split()]
    red_balls.sort()  # 排序，便于后续分析
    
    # 1. 和值
    sum_value = sum(red_balls)
    
    # 2. 平均值
    avg_value = sum_value / len(red_balls)
    
    # 3. 尾数和值
    tail_sum = sum([x % 10 for x in red_balls])
    
    # 4. 奇偶分析
    odd_nums = len([x for x in red_balls if x % 2 == 1])
    even_nums = len(red_balls) - odd_nums
    odd_even_diff = odd_nums - even_nums
    
    # 5. 奇偶连续
    odd_continuous = 0
    even_continuous = 0
    current_odd = 0
    current_even = 0
    
    for num in red_balls:
        if num % 2 == 1:  # 奇数
            current_odd += 1
            current_even = 0
            odd_continuous = max(odd_continuous, current_odd)
        else:  # 偶数
            current_even += 1
            current_odd = 0
            even_continuous = max(even_continuous, current_even)
    
    # 6. 大小号分析（以16为界）
    big_nums = len([x for x in red_balls if x > 16])
    small_nums = len(red_balls) - big_nums
    big_small_diff = big_nums - small_nums
    
    # 7. 尾号组数
    tail_groups = len(set([x % 10 for x in red_balls]))
    
    # 8. AC值（任意两个号码之差的绝对值的和）
    ac_value = sum(abs(a - b) for a, b in combinations(red_balls, 2))
    
    # 9. 连号分析
    consecutive_count = 0  # 连号个数
    consecutive_groups = 0  # 连号组数
    for i in range(len(red_balls) - 1):
        if red_balls[i + 1] - red_balls[i] == 1:
            consecutive_count += 1
            if i == 0 or red_balls[i] - red_balls[i - 1] > 1:
                consecutive_groups += 1
    
    # 10. 首尾差
    span = red_balls[-1] - red_balls[0]
    
    # 11. 最大间距
    max_gap = max([red_balls[i+1] - red_balls[i] for i in range(len(red_balls)-1)])
    
    # 12. 重号个数（与上一期相比，需要外部传入上一期数据）
    repeat_nums = 0  # 这个值需要在主程序中计算
    
    # 13. 斜号个数（与上一期相比的差值为1或-1的个数，需要外部传入上一期数据）
    diagonal_nums = 0  # 这个值需要在主程序中计算
    
    return {
        '和值': sum_value,
        '平均值': round(avg_value, 2),
        '尾数和值': tail_sum,
        '奇号个数': odd_nums,
        '偶号个数': even_nums,
        '奇偶偏差': odd_even_diff,
        '奇号连续': odd_continuous,
        '偶号连续': even_continuous,
        '大号个数': big_nums,
        '小号个数': small_nums,
        '大小偏差': big_small_diff,
        '尾号组数': tail_groups,
        'AC值': ac_value,
        '连号个数': consecutive_count,
        '连号组数': consecutive_groups,
        '首尾差': span,
        '最大间距': max_gap,
        '重号个数': repeat_nums,
        '斜号个数': diagonal_nums
    }

def compare_with_previous(current_numbers, previous_numbers):
    """比较当前号码与上一期号码的重号和斜号"""
    if not previous_numbers:
        return 0, 0
        
    current = set(int(x) for x in current_numbers.split())
    previous = set(int(x) for x in previous_numbers.split())
    
    # 计算重号个数
    repeat_count = len(current & previous)
    
    # 计算斜号个数
    diagonal_count = 0
    for num in previous:
        if (num + 1) in current or (num - 1) in current:
            diagonal_count += 1
            
    return repeat_count, diagonal_count 