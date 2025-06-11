import os
import sys
from datetime import datetime
import pandas as pd
from ssq_crawler import (
    fetch_ssq_data,
    load_data,
    display_lottery_info,
    analyze_lottery_data,
    initialize_data,
    display_comprehensive_recommendations,
)
from number_predictor import NumberPredictor
from ssq_analysis import analyze_red_balls, compare_with_previous

class SSQManager:
    def __init__(self):
        self.data = None
        self.predictor = None
        self.initialize()
    
    def initialize(self):
        """初始化数据和预测器"""
        print("正在初始化系统...")

        # 优先尝试自动更新（包含新旧数据检查）
        self.data = initialize_data()

        # 若自动更新失败，则回退到本地数据
        if self.data is None:
            print("自动更新失败，尝试加载本地数据...")
            self.data = load_data()
            # 若本地数据缺少分析列则补充
            if self.data is not None and '和值' not in self.data.columns:
                try:
                    self.data = analyze_lottery_data(self.data)
                except Exception as e:
                    print(f"分析数据时发生错误: {e}")
                    self.data = None

        if self.data is not None:
            self.predictor = NumberPredictor(self.data)
            print("系统初始化完成！")
        else:
            print("初始化失败，请检查网络连接后重试。")
            sys.exit(1)
    
    def update_data(self):
        """更新开奖数据"""
        print("正在更新开奖数据...")
        new_data = fetch_ssq_data()
        if new_data is not None:
            self.data = new_data
            self.predictor = NumberPredictor(self.data)
            print("数据更新成功！")
        else:
            print("数据更新失败，将继续使用现有数据。")
    
    def show_latest_draw(self):
        """显示最新一期开奖信息"""
        display_lottery_info(self.data)
    
    def show_simple_latest_draw(self):
        """仅显示最新一期的开奖信息（不含分析）"""
        if self.data is None or self.data.empty:
            print("暂无开奖数据，请先更新。")
            return

        latest = self.data.iloc[0]
        red_str = ' '.join(f"{int(n):02d}" for n in latest['红球'].split())
        print(f"期号：{latest['期号']}  开奖日期：{latest['开奖日期']}  中奖号码：红球 {red_str} + 蓝球 {int(latest['蓝球']):02d}")
    
    def search_historical_draw(self, query):
        """搜索历史开奖信息"""
        display_lottery_info(self.data, query)
    
    def analyze_trends(self, periods=10):
        """分析近期走势"""
        if periods > len(self.data):
            periods = len(self.data)
        
        recent_data = self.data.head(periods)
        print(f"\n=== 近{periods}期走势分析 ===")
        
        # 红球分析
        red_analysis = []
        for _, row in recent_data.iterrows():
            analysis = analyze_red_balls(row['红球'])
            analysis['期号'] = row['期号']
            red_analysis.append(analysis)
        
        df_analysis = pd.DataFrame(red_analysis)
        
        # 显示关键指标
        print("\n红球指标统计：")
        print(f"和值范围：{df_analysis['和值'].min()}-{df_analysis['和值'].max()}")
        print(f"奇偶比例：{df_analysis['奇号个数'].mean():.1f}:{df_analysis['偶号个数'].mean():.1f}")
        print(f"大小比例：{df_analysis['大号个数'].mean():.1f}:{df_analysis['小号个数'].mean():.1f}")
        print(f"平均连号数：{df_analysis['连号个数'].mean():.1f}")
        
        # 蓝球分析
        blue_numbers = recent_data['蓝球'].value_counts()
        print("\n蓝球出现频率（前5名）：")
        for num, count in blue_numbers.head().items():
            print(f"蓝球 {int(num):02d}: {count}次")
    
    def get_recommendations(self, num_combinations=5):
        """获取号码推荐"""
        print(f"\n=== 号码推荐（{num_combinations}注） ===")
        recommendations = self.predictor.generate_numbers(num_combinations)
        
        for i, (reds, blue) in enumerate(recommendations, 1):
            red_str = ' '.join(f"{n:02d}" for n in sorted(reds))
            print(f"第{i}注：红球 {red_str} + 蓝球 {blue:02d}")
            
            # 显示号码分析
            analysis = self.predictor.get_number_analysis((reds, blue))
            print("    号码分析：")
            print(f"    和值：{analysis['和值']}")
            print(f"    奇偶比：{analysis['奇号个数']}:{analysis['偶号个数']}")
            print(f"    大小比：{analysis['大号个数']}:{analysis['小号个数']}")
            print(f"    连号个数：{analysis['连号个数']}个")
            print(f"    AC值：{analysis['AC值']}")

            # 综合概率评分
            prob_analysis = self.predictor.analyze_winning_probability((reds, blue))
            print(f"    综合中奖概率评分：{prob_analysis['probability']:.2f}%")

            details = prob_analysis['score_details']
            print("    详细评分：")
            print(f"      1. 和值合理性：{'✓' if details['和值合理性'] else '✗'}")
            print(f"      2. 间隔合理性：{'✓' if details['间隔合理性'] else '✗'}")
            print(f"      3. 区域分布得分：{details['区域分布']:.2f}/1.00")
            print(f"      4. 质数比例偏差：{details['质数比例']:.2f}")
            print(f"      5. 连号个数：{details['连号合理性']} 组")
            print(f"      6. 遗漏值分析得分：{details['遗漏值分析']:.2f}/1.00")
            print(f"      7. 蓝球热门：{'✓' if details['蓝球可能性'] else '✗'}")
            print()
    
    def show_menu(self):
        """显示主菜单"""
        while True:
            # 显示最新开奖信息（简洁版）
            print("\n=== 最新开奖信息 ===")
            self.show_simple_latest_draw()

            print("\n=== 双色球助手 ===")
            print("1. 查看最新开奖")
            print("2. 搜索历史开奖")
            print("3. 分析近期走势")
            print("4. 号码推荐（单式/复式）")
            print("5. 更新开奖数据")
            print("0. 退出程序")
            
            choice = input("\n请选择功能（0-5）：")
            
            if choice == '1':
                self.show_latest_draw()
            elif choice == '2':
                query = input("请输入期号或日期（如：23001 或 2023-01-01）：")
                self.search_historical_draw(query)
            elif choice == '3':
                try:
                    periods = int(input("请输入要分析的期数（默认10期）：") or "10")
                    self.analyze_trends(periods)
                except ValueError:
                    print("输入无效，将使用默认值10期")
                    self.analyze_trends()
            elif choice == '4':
                # 综合号码推荐（内含单式和复式选项）
                display_comprehensive_recommendations(self.predictor)
            elif choice == '5':
                self.update_data()
            elif choice == '0':
                print("感谢使用，再见！")
                break
            else:
                print("无效的选择，请重试。")

def main():
    """主函数"""
    try:
        manager = SSQManager()
        manager.show_menu()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序发生错误：{e}")
    finally:
        print("\n程序已退出")

if __name__ == "__main__":
    main() 