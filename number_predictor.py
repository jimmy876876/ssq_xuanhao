import pandas as pd
import numpy as np
from collections import Counter
from itertools import combinations
import math
import random

class NumberPredictor:
    def __init__(self, analysis_df):
        self.df = analysis_df
        self.red_numbers = range(1, 34)  # 红球号码范围
        self.blue_numbers = range(1, 17)  # 蓝球号码范围
        self.analyze_historical_patterns()  # 初始化时分析历史规律
        
    def analyze_historical_patterns(self):
        """分析历史数据规律"""
        self.patterns = {
            'sum_range': self._analyze_sum_range(),
            'number_gaps': self._analyze_number_gaps_pattern(),
            'position_frequency': self._analyze_position_frequency(),
            'missing_values': self._analyze_missing_values(),
            'zone_distribution': self._analyze_zone_distribution(),
            'prime_composite': self._analyze_prime_composite_ratio(),
            'consecutive_patterns': self._analyze_consecutive_patterns(),
            'winning_patterns': self._analyze_winning_patterns(),  # 新增中奖号码模式分析
            'number_trends': self._analyze_number_trends(),  # 新增号码走势分析
            'adjacent_sum': self._analyze_adjacent_sum(),  # 新增相邻号码和值分析
            'last_digits': self._analyze_last_digits(),  # 新增尾数分析
            'sum_tail': self._analyze_sum_tail()  # 新增和值尾数分析
        }
    
    def _analyze_sum_range(self):
        """分析和值范围分布"""
        sums = []
        for numbers in self.df['红球']:
            nums = [int(x) for x in numbers.split()]
            sums.append(sum(nums))
        
        return {
            'min': min(sums),
            'max': max(sums),
            'mean': np.mean(sums),
            'std': np.std(sums),
            'most_common': Counter(sums).most_common(3)
        }
    
    def _analyze_number_gaps_pattern(self):
        """分析号码间隔模式"""
        gaps_patterns = []
        for numbers in self.df['红球']:
            nums = sorted([int(x) for x in numbers.split()])
            gaps = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
            gaps_patterns.append(tuple(gaps))
        
        return {
            'common_patterns': Counter(gaps_patterns).most_common(5),
            'avg_gaps': np.mean([sum(gap)/len(gap) for gap in gaps_patterns])
        }
    
    def _analyze_position_frequency(self):
        """分析不同位置上号码的出现频率"""
        position_numbers = {i: [] for i in range(6)}  # 6个位置
        
        for numbers in self.df['红球']:
            nums = sorted([int(x) for x in numbers.split()])
            for pos, num in enumerate(nums):
                position_numbers[pos].append(num)
        
        return {pos: Counter(nums).most_common(5) for pos, nums in position_numbers.items()}
    
    def _analyze_missing_values(self):
        """分析号码遗漏值"""
        missing_values = {num: 0 for num in range(1, 34)}
        last_appearance = {num: -1 for num in range(1, 34)}
        current_draw = 0
        
        for numbers in self.df['红球']:
            nums = [int(x) for x in numbers.split()]
            for num in range(1, 34):
                if num in nums:
                    missing_values[num] = max(missing_values[num], current_draw - last_appearance[num])
                    last_appearance[num] = current_draw
            current_draw += 1
        
        return {
            'max_missing': {num: val for num, val in missing_values.items() if val > 0},
            'current_missing': {num: current_draw - last_appearance[num] for num in range(1, 34)}
        }
    
    def _analyze_zone_distribution(self):
        """分析号码区域分布"""
        zones = {
            'low': (1, 11),
            'mid': (12, 22),
            'high': (23, 33)
        }
        
        zone_counts = []
        for numbers in self.df['红球']:
            nums = [int(x) for x in numbers.split()]
            count = {zone: len([n for n in nums if zones[zone][0] <= n <= zones[zone][1]]) 
                    for zone in zones}
            zone_counts.append(count)
        
        return {
            'average': {zone: np.mean([c[zone] for c in zone_counts]) for zone in zones},
            'most_common': {zone: Counter([c[zone] for c in zone_counts]).most_common(1)[0] 
                          for zone in zones}
        }
    
    def _is_prime(self, n):
        """判断是否为质数"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _analyze_prime_composite_ratio(self):
        """分析质数和合数比例"""
        ratios = []
        for numbers in self.df['红球']:
            nums = [int(x) for x in numbers.split()]
            prime_count = len([n for n in nums if self._is_prime(n)])
            ratios.append(prime_count / len(nums))
        
        return {
            'average_ratio': np.mean(ratios),
            'most_common': Counter([round(r, 2) for r in ratios]).most_common(3)
        }
    
    def _analyze_consecutive_patterns(self):
        """分析连号模式"""
        patterns = []
        for numbers in self.df['红球']:
            nums = sorted([int(x) for x in numbers.split()])
            consecutive = []
            current = []
            
            for i in range(len(nums)-1):
                if nums[i+1] - nums[i] == 1:
                    if not current:
                        current = [nums[i]]
                    current.append(nums[i+1])
                else:
                    if current:
                        consecutive.append(tuple(current))
                        current = []
            
            if current:
                consecutive.append(tuple(current))
            
            patterns.append(tuple(consecutive))
        
        return {
            'common_patterns': Counter(patterns).most_common(5),
            'avg_consecutive_groups': np.mean([len(p) for p in patterns])
        }
    
    def _analyze_winning_patterns(self):
        """分析中奖号码的特征模式"""
        patterns = {
            'repeat_count': [],  # 重复号码数量
            'gap_patterns': [],  # 间隔模式
            'sum_patterns': [],  # 和值模式
            'position_patterns': [],  # 位置模式
            'zone_patterns': []  # 区域分布模式
        }
        
        for i in range(len(self.df) - 1):
            current = [int(x) for x in self.df.iloc[i]['红球'].split()]
            previous = [int(x) for x in self.df.iloc[i + 1]['红球'].split()]
            
            # 分析重复号码
            repeat = len(set(current) & set(previous))
            patterns['repeat_count'].append(repeat)
            
            # 分析间隔模式
            gaps = [current[j+1] - current[j] for j in range(len(current)-1)]
            patterns['gap_patterns'].append(tuple(gaps))
            
            # 分析和值模式
            sum_val = sum(current)
            patterns['sum_patterns'].append(sum_val)
            
            # 分析位置模式
            position_pattern = []
            for j, num in enumerate(current):
                if j > 0:
                    if num > current[j-1]:
                        position_pattern.append(1)
                    else:
                        position_pattern.append(-1)
            patterns['position_patterns'].append(tuple(position_pattern))
            
            # 分析区域分布
            zones = {'low': 0, 'mid': 0, 'high': 0}
            for num in current:
                if num <= 11:
                    zones['low'] += 1
                elif num <= 22:
                    zones['mid'] += 1
                else:
                    zones['high'] += 1
            patterns['zone_patterns'].append((zones['low'], zones['mid'], zones['high']))
        
        return {
            'repeat_stats': {
                'avg': np.mean(patterns['repeat_count']),
                'most_common': Counter(patterns['repeat_count']).most_common(3)
            },
            'gap_stats': {
                'most_common': Counter(patterns['gap_patterns']).most_common(5)
            },
            'sum_stats': {
                'mean': np.mean(patterns['sum_patterns']),
                'std': np.std(patterns['sum_patterns']),
                'range': (min(patterns['sum_patterns']), max(patterns['sum_patterns']))
            },
            'position_stats': {
                'most_common': Counter(patterns['position_patterns']).most_common(5)
            },
            'zone_stats': {
                'most_common': Counter(patterns['zone_patterns']).most_common(5)
            }
        }
    
    def _analyze_number_trends(self):
        """分析号码走势"""
        trends = {num: [] for num in range(1, 34)}
        window_size = 10  # 分析最近10期的走势
        
        for i in range(min(window_size, len(self.df))):
            numbers = set(int(x) for x in self.df.iloc[i]['红球'].split())
            for num in range(1, 34):
                trends[num].append(1 if num in numbers else 0)
        
        trend_analysis = {}
        for num in range(1, 34):
            # 计算趋势强度
            trend_strength = sum(trends[num])
            # 计算最近出现距离
            last_appearance = trends[num].index(1) if 1 in trends[num] else window_size
            # 计算周期性（如果有）
            cycle = self._find_cycle(trends[num]) if sum(trends[num]) > 1 else 0
            
            trend_analysis[num] = {
                'strength': trend_strength,
                'last_appearance': last_appearance,
                'cycle': cycle
            }
        
        return trend_analysis
    
    def _find_cycle(self, sequence):
        """查找序列中的周期性"""
        if len(sequence) < 4:
            return 0
        
        for length in range(2, len(sequence)//2 + 1):
            is_cycle = True
            for i in range(len(sequence) - length):
                if sequence[i] != sequence[i + length]:
                    is_cycle = False
                    break
            if is_cycle:
                return length
        return 0
    
    def _analyze_adjacent_sum(self):
        """分析相邻号码和值特征"""
        adjacent_sums = []
        for numbers in self.df['红球']:
            nums = sorted([int(x) for x in numbers.split()])
            sums = [nums[i] + nums[i+1] for i in range(len(nums)-1)]
            adjacent_sums.append(sums)
        
        return {
            'mean': np.mean([sum(s)/len(s) for s in adjacent_sums]),
            'std': np.std([sum(s)/len(s) for s in adjacent_sums]),
            'patterns': Counter([tuple(s) for s in adjacent_sums]).most_common(5)
        }
    
    def _analyze_last_digits(self):
        """分析尾数特征"""
        last_digits = []
        for numbers in self.df['红球']:
            digits = [int(x) % 10 for x in numbers.split()]
            last_digits.append(digits)
        
        return {
            'frequency': Counter([d for digits in last_digits for d in digits]),
            'patterns': Counter([tuple(sorted(digits)) for digits in last_digits]).most_common(5)
        }
    
    def _analyze_sum_tail(self):
        """分析和值尾数特征"""
        sum_tails = []
        for numbers in self.df['红球']:
            nums = [int(x) for x in numbers.split()]
            sum_tail = sum(nums) % 10
            sum_tails.append(sum_tail)
        
        return {
            'frequency': Counter(sum_tails),
            'most_common': Counter(sum_tails).most_common(3)
        }
    
    def analyze_winning_probability(self, numbers):
        """分析号码组合的中奖概率"""
        red_numbers, blue_ball = numbers
        score = 0
        max_score = 10
        
        # 1. 和值分析
        sum_val = sum(red_numbers)
        if (self.patterns['sum_range']['mean'] - self.patterns['sum_range']['std'] <= 
            sum_val <= 
            self.patterns['sum_range']['mean'] + self.patterns['sum_range']['std']):
            score += 1
        
        # 2. 间隔分析
        gaps = [red_numbers[i+1] - red_numbers[i] for i in range(len(red_numbers)-1)]
        avg_gap = sum(gaps) / len(gaps)
        if abs(avg_gap - self.patterns['number_gaps']['avg_gaps']) < 2:
            score += 1
        
        # 3. 区域分布分析
        zones = {
            'low': len([n for n in red_numbers if 1 <= n <= 11]),
            'mid': len([n for n in red_numbers if 12 <= n <= 22]),
            'high': len([n for n in red_numbers if 23 <= n <= 33])
        }
        
        zone_score = sum(1 for zone in zones if abs(zones[zone] - 
                        self.patterns['zone_distribution']['average'][zone]) < 1)
        score += zone_score / 3
        
        # 4. 质数比例分析
        prime_count = len([n for n in red_numbers if self._is_prime(n)])
        prime_ratio = prime_count / len(red_numbers)
        if abs(prime_ratio - self.patterns['prime_composite']['average_ratio']) < 0.2:
            score += 1
        
        # 5. 连号分析
        consecutive = []
        current = []
        for i in range(len(red_numbers)-1):
            if red_numbers[i+1] - red_numbers[i] == 1:
                if not current:
                    current = [red_numbers[i]]
                current.append(red_numbers[i+1])
            else:
                if current:
                    consecutive.append(tuple(current))
                    current = []
        if current:
            consecutive.append(tuple(current))
        
        if len(consecutive) <= self.patterns['consecutive_patterns']['avg_consecutive_groups'] + 1:
            score += 1
        
        # 6. 遗漏值分析
        missing_score = 0
        for num in red_numbers:
            current_missing = self.patterns['missing_values']['current_missing'][num]
            if current_missing > 5:  # 遗漏值大于5期的号码
                missing_score += 0.2
        score += min(missing_score, 1)
        
        # 7. 蓝球分析
        recent_blues = self.analyze_recent_blue_balls()
        if blue_ball in recent_blues[:5]:  # 如果蓝球在最近的高频号码中
            score += 1
        
        # 计算最终概率
        probability = (score / max_score) * 100
        
        return {
            'probability': probability,
            'score_details': {
                '和值合理性': score >= 1,
                '间隔合理性': score >= 2,
                '区域分布': zone_score / 3,
                '质数比例': abs(prime_ratio - self.patterns['prime_composite']['average_ratio']),
                '连号合理性': len(consecutive),
                '遗漏值分析': missing_score,
                '蓝球可能性': blue_ball in recent_blues[:5]
            }
        }
    
    def generate_complex_numbers(self, strategy='balanced'):
        """生成复式投注推荐号码"""
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
        red_count = random.randint(*config['red'])
        
        # 候选号码池
        candidates = []
        
        # 1. 从历史高频号码中选择
        hot_numbers, _ = self.analyze_hot_cold_numbers()
        candidates.extend(random.sample(hot_numbers, min(len(hot_numbers), int(red_count * 0.4))))
        
        # 2. 从最近遗漏值较大的号码中选择
        missing_numbers = sorted(self.patterns['missing_values']['current_missing'].items(),
                               key=lambda x: x[1], reverse=True)
        missing_candidates = [num for num, _ in missing_numbers[:10] if num not in candidates]
        candidates.extend(random.sample(missing_candidates, min(len(missing_candidates), int(red_count * 0.3))))
        
        # 3. 根据区域分布补充号码
        zones = {
            'low': (1, 11),
            'mid': (12, 22),
            'high': (23, 33)
        }
        
        for zone, (start, end) in zones.items():
            target_count = math.ceil(self.patterns['zone_distribution']['average'][zone])
            current_count = len([n for n in candidates if start <= n <= end])
            if current_count < target_count:
                zone_numbers = [n for n in range(start, end + 1) if n not in candidates]
                if zone_numbers:
                    candidates.extend(random.sample(zone_numbers, min(len(zone_numbers), target_count - current_count)))
        
        # 4. 随机补充剩余号码
        remaining_count = red_count - len(candidates)
        if remaining_count > 0:
            remaining_numbers = [n for n in range(1, 34) if n not in candidates]
            candidates.extend(random.sample(remaining_numbers, remaining_count))
        
        # 选择蓝球
        blue_count = random.randint(*config['blue'])
        recent_blues = self.analyze_recent_blue_balls()
        blue_candidates = set(random.sample(recent_blues[:8], min(len(recent_blues[:8]), int(blue_count * 0.7))))
        
        remaining_blue_count = blue_count - len(blue_candidates)
        if remaining_blue_count > 0:
            remaining_blues = [n for n in range(1, 17) if n not in blue_candidates]
            blue_candidates.update(random.sample(remaining_blues, remaining_blue_count))
        
        return sorted(candidates), sorted(blue_candidates)
        
    def analyze_hot_cold_numbers(self, recent_periods=30):
        """分析冷热号"""
        recent_data = self.df.head(recent_periods)
        all_numbers = []
        for numbers in recent_data['红球']:
            all_numbers.extend([int(x) for x in numbers.split()])
        
        # 统计每个号码出现的次数
        number_freq = Counter(all_numbers)
        
        # 将号码分为冷热号
        avg_freq = len(all_numbers) / 33  # 平均出现频率
        hot_numbers = [num for num, freq in number_freq.items() if freq > avg_freq]
        cold_numbers = [num for num in self.red_numbers if num not in hot_numbers]
        
        return hot_numbers, cold_numbers
    
    def analyze_number_gaps(self):
        """分析号码间隔"""
        gaps = []
        for numbers in self.df['红球']:
            nums = sorted([int(x) for x in numbers.split()])
            gaps.extend([nums[i+1] - nums[i] for i in range(len(nums)-1)])
        
        avg_gap = np.mean(gaps)
        return avg_gap
    
    def analyze_sum_value_distribution(self):
        """分析和值分布"""
        sum_values = self.df['和值']
        avg_sum = sum_values.mean()
        std_sum = sum_values.std()
        return avg_sum, std_sum
    
    def analyze_odd_even_pattern(self):
        """分析奇偶比例"""
        odd_counts = self.df['奇号个数']
        most_common_odd = odd_counts.mode().iloc[0]
        return most_common_odd
    
    def analyze_big_small_pattern(self):
        """分析大小比例"""
        big_counts = self.df['大号个数']
        most_common_big = big_counts.mode().iloc[0]
        return most_common_big
    
    def analyze_consecutive_numbers(self):
        """分析连号情况"""
        consecutive_counts = self.df['连号个数']
        avg_consecutive = consecutive_counts.mean()
        return avg_consecutive
    
    def analyze_blue_ball_frequency(self, recent_periods=50):
        """分析蓝球出现频率"""
        recent_data = self.df.head(recent_periods)
        blue_numbers = [int(x) for x in recent_data['蓝球']]
        blue_freq = Counter(blue_numbers)
        return blue_freq
    
    def analyze_recent_blue_balls(self, recent_periods=30):
        """分析最近期数的蓝球出现情况，返回按出现频率排序的蓝球列表"""
        blue_freq = self.analyze_blue_ball_frequency(recent_periods)
        # 按出现频率排序
        sorted_blues = sorted(blue_freq.items(), key=lambda x: (-x[1], x[0]))  # 频率降序，号码升序
        return [num for num, _ in sorted_blues]
    
    def generate_numbers(self, num_combinations=5):
        """生成推荐号码（增强版）"""
        recommended_combinations = []
        
        for _ in range(num_combinations):
            red_numbers = self._generate_optimized_reds()
            blue_ball = self._select_optimal_blue()
            recommended_combinations.append((red_numbers, blue_ball))
        
        return recommended_combinations
    
    def _generate_optimized_reds(self):
        """生成优化的红球号码"""
        candidates = set()
        
        # 1. 根据历史中奖模式选择基础号码
        winning_patterns = self.patterns['winning_patterns']
        target_sum = int(winning_patterns['sum_stats']['mean'])
        target_zone = max(winning_patterns['zone_stats']['most_common'], key=lambda x: x[1])[0]
        
        # 2. 考虑号码走势
        trends = self.patterns['number_trends']
        trending_numbers = sorted(
            [(num, data) for num, data in trends.items()],
            key=lambda x: (-x[1]['strength'], x[1]['last_appearance'])
        )
        
        # 选择2-3个走势强的号码
        strong_trends = [num for num, _ in trending_numbers[:8]]
        candidates.update(set(random.sample(strong_trends, random.randint(2, 3))))
        
        # 3. 考虑遗漏值
        missing_values = self.patterns['missing_values']['current_missing']
        high_missing = sorted(
            [(num, val) for num, val in missing_values.items()],
            key=lambda x: x[1], reverse=True
        )
        
        # 选择1-2个遗漏值高的号码
        high_missing_nums = [num for num, _ in high_missing[:10]]
        candidates.update(set(random.sample(high_missing_nums, random.randint(1, 2))))
        
        # 4. 根据区域分布补充号码
        zones = {'low': (1, 11), 'mid': (12, 22), 'high': (23, 33)}
        target_distribution = target_zone
        
        for zone, (start, end) in zones.items():
            target_count = target_distribution[{'low': 0, 'mid': 1, 'high': 2}[zone]]
            current_count = len([n for n in candidates if start <= n <= end])
            if current_count < target_count:
                zone_numbers = [n for n in range(start, end + 1) 
                              if n not in candidates]
                if zone_numbers:
                    candidates.update(set(random.sample(
                        zone_numbers,
                        min(len(zone_numbers), target_count - current_count)
                    )))
        
        # 5. 补充剩余号码，确保和值接近目标值
        while len(candidates) < 6:
            current_sum = sum(candidates)
            if current_sum < target_sum:
                # 选择较大的号码
                available = [n for n in range(23, 34) if n not in candidates]
            else:
                # 选择较小的号码
                available = [n for n in range(1, 12) if n not in candidates]
            
            if available:
                candidates.add(random.choice(available))
        
        # 6. 优化号码组合
        candidates = list(candidates)
        # 确保符合相邻和值特征
        adj_sum_mean = self.patterns['adjacent_sum']['mean']
        for _ in range(10):  # 尝试优化10次
            sorted_nums = sorted(candidates)
            adj_sums = [sorted_nums[i] + sorted_nums[i+1] for i in range(len(sorted_nums)-1)]
            current_adj_mean = sum(adj_sums) / len(adj_sums)
            
            if abs(current_adj_mean - adj_sum_mean) <= 2:
                break
                
            # 替换一个号码
            idx = random.randint(0, 5)
            candidates.remove(sorted_nums[idx])
            available = [n for n in range(1, 34) if n not in candidates]
            candidates.append(random.choice(available))
        
        return sorted(candidates)
    
    def _select_optimal_blue(self):
        """选择最优蓝球"""
        recent_blues = self.analyze_recent_blue_balls()
        weights = [1.0 / (i + 1) for i in range(len(recent_blues))]  # 赋予更近期的号码更高权重
        
        # 综合考虑最近出现频率和历史规律
        if random.random() < 0.7:  # 70%概率选择最近的高频号码
            return random.choices(recent_blues[:5], weights[:5], k=1)[0]
        else:  # 30%概率选择其他号码
            return random.randint(1, 16)
    
    def get_number_analysis(self, numbers):
        """分析生成的号码组合"""
        red_numbers, blue_ball = numbers
        analysis = {
            '和值': sum(red_numbers),
            '奇号个数': len([x for x in red_numbers if x % 2 == 1]),
            '偶号个数': len([x for x in red_numbers if x % 2 == 0]),
            '大号个数': len([x for x in red_numbers if x > 16]),
            '小号个数': len([x for x in red_numbers if x <= 16]),
            '连号个数': sum(1 for i in range(len(red_numbers)-1) if red_numbers[i+1] - red_numbers[i] == 1),
            'AC值': sum(abs(a - b) for a, b in combinations(red_numbers, 2))
        }
        return analysis 