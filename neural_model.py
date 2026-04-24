import math

# ���� � ��������� ��������� (�� ������ C-����)
input_hidden_weights = [
    [4.11248014787830, -0.880730724262271, -0.282759441554233, 2.67806209371124],
    [-2.90793602890967, -0.995542519747002, 2.70207046407578, -3.98743926461036],
    [2.38265168288832, 2.68525858484784, -1.51788803647873, 0.586348080650260],
    [-1.15644716355043, 0.296900922868515, -1.01101313683287, 0.967198092342738]
]

hidden_bias = [-0.756309737876200, 0.446097858172558, -1.01818563548482, -0.176428537776959]

hidden_output_weights = [
    [1.76303798240094, 1.76609252603824, 1.18745308708255, 2.80804465019565]
]

output_bias = [1.85139685601618]

max_input = [588.0, 936.0, 243.0, 208.0]
min_input = [86.0,  190.0,  84.0,  74.0]
max_target = [974.0]
min_target = [145.0]
mean_inputs = [283.46, 484.46, 155.12, 117.72]


def scale_inputs(inputs):
    scaled = []
    for i in range(4):
        delta = 1.0 / (max_input[i] - min_input[i])
        val = -delta * min_input[i] + delta * inputs[i]
        scaled.append(val)
    return scaled


def unscale_output(output_val):
    delta = 1.0 / (max_target[0] - min_target[0])
    return (output_val - delta * min_target[0]) / delta


def tanh(x):
    return math.tanh(x)


def predict(inp):
    # ����������� ����������� ��������
    for i in range(4):
        if inp[i] == -9999:
            inp[i] = mean_inputs[i]

    # ��������������� ������
    scaled = scale_inputs(inp)

    # ������� ����
    hidden = []
    for row in range(4):
        s = sum(input_hidden_weights[row][col] * scaled[col] for col in range(4))
        s += hidden_bias[row]
        hidden.append(tanh(s))

    # �������� ����
    out = sum(hidden_output_weights[0][col] * hidden[col] for col in range(4))
    out += output_bias[0]
    out = tanh(out)

    # �������� ���������������
    result = unscale_output(out)
    return round(result, 2)