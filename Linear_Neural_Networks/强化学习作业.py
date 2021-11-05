import numpy as np


def get_state_values(is_average_policy=1):
    """
    :param is_average_policy: 是否选择平均策略 1代表平均策略 0代表贪心策略
    :return:
    """
    # 状态一共有16个,第一个和最后一个状态我们认为是出口,对其v值不做改动
    state_space = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    # 平均策略和贪心策略状态的初始值和策略函数不同
    if is_average_policy == 1:
        print("你选择了平均策略----")
        # state_values = np.zeros((4, 4))
        state_values = np.asarray([[0, -1, -2, -3],
                                   [-1, -2, -3, -2],
                                   [-2, -3, -2, -1],
                                   [-3, -2, -1, 0]])
        policy_values = np.full((16, 4), 0.25)

    else:
        print("你选择了贪心策略----")
        # state_values = np.zeros((4, 4))
        state_values = np.asarray([[0, -14, -20, -22],
                                   [-14, -18, -20, -20],
                                   [-20, -20, -18, -14],
                                   [-22, -20, -14, 0]])

        policy_values = np.zeros((16, 4))
    # 我们用一个列表存储当前v值和下一个v值,当两个v值相等,我们就认为已经收敛,跳出循环
    # 把下一个v值初始化为全0的4*4张量
    new_state_values = np.zeros((4, 4))
    # 创建value_list存储两个v值
    value_list = [state_values, new_state_values]
    # 用i记录迭代次数
    i = 0
    while True:
        # 这里如果为贪心策略 每次迭代要初始化策略函数 不然会发生错误
        if is_average_policy == 0:
            policy_values = np.zeros((16, 4))
        # 迭代次数需要限制 在1000次循环之后 跳出循环 返回当前的v值作为最终的v值
        if i >= 1000:
            print("该策略在经过1000次迭代后仍未收敛,此时的v值为:", value_list[0])
            break
        # 这个for循环代表一次的迭代过程,把每个状态对应的v值分别改变,然后存入value_list中
        for j in state_space:
            # 对于入口和出口 v值不做改变
            if 0 < j < 15:
                # print("policy_values", policy_values)
                # 我们的思路是v=pai*(r+p*v) 我们把每一个状态对应的pai和(r+p*v)表示成向量 然后对向量做点积
                # 由于问题被简化 所以说选择任何一个状态所导致的结果是固定的(即下一个状态是固定的)也就是说p为1,显然这里r为-1
                # 这里value_list[0]表示上一个状态 value_list[1]表示下一个状态 是要求的值
                # action_list 表示选择不同策略得到的下个状态的v值 即上下左右移动后得到的v值 我们需要给v值加上一个reward 也就是-1
                action_list = [
                    value_list[0][j // 4 - 1 if j // 4 > 0 else j // 4, j % 4],  # 上
                    value_list[0][j // 4 + 1 if j // 4 < 3 else j // 4, j % 4],  # 下
                    value_list[0][j // 4, j % 4 - 1 if j % 4 > 0 else j % 4],  # 左
                    value_list[0][j // 4, j % 4 + 1 if j % 4 < 3 else j % 4]]  # 右
                # 把list转化为张量 帮助我们计算
                action_list = np.asarray(action_list)
                # 这里需要注意 如果我们选择贪心策略的话 策略函数policy_values的值并不是固定的 与下一个状态的最大v值有关
                # 因此我们先把policy_values全部初始化为0 选择最大的下个v值对应的策略 把该策略置为1 其他仍然是0
                if is_average_policy == 0:
                    # 找到最大的策略对应的下标  这里遇到重复的策略值 只能返回第一个重复值
                    index = np.argmax(action_list)
                    # 把该行对应元素的pai值置为1
                    policy_values[j, index] = 1
                # 对于每一个状态 我们都算出该状态的下一个v值 不论是否为贪心策略 用value_list[1]存储
                value_list[1][j // 4, j % 4] = np.dot(action_list - 1, policy_values[j])
        # 判断是否收敛 这个判断一定要在一次迭代完成之后,即在for循环外边 不要放错位置
        if (value_list[0] == value_list[1]).all():
            print(f"该策略在经过{i}次之后收敛")
            print("收敛值为:", value_list[1])
            # 当策略为贪心策略时  给出具体路径
            if is_average_policy == 0:
                direction = ["上", "下", "左", "右"]
                policy_list = [[], [], [], []]
                print("最优策略值为:", policy_values)
                for i, policy_value in enumerate(policy_values):
                    direction_index = np.argmax(policy_value)
                    if i == 0:
                        policy_list[i // 4].append("入口")
                    elif i == 15:
                        policy_list[i // 4].append("出口")
                    else:
                        policy_list[i // 4].append(direction[direction_index])
                print("最优策略为:", policy_list)
            break
        # 对于每一个状态 在算出下个v值后 我们都应该更新value_list 同样 更新状态也应该在for循环外边
        value_list[0] = value_list[1]
        # 这里必须重置0 不然value_list[0]和value_list[1]就会指向同一个内存
        # 下次改动value_list[1]的时候,value_list[0]也会被改变
        value_list[1] = np.zeros((4, 4))
        # 最后迭代次数加一
        i = i + 1


if __name__ == '__main__':
    # 1代表平均策略 0代表贪心策略
    get_state_values(is_average_policy=1)

