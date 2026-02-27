from Forge import Array, forge


@forge(debug=True)
def fn(weights, biases, input):
    act = input
    for i in range(2):
        act = weights[i] @ act + biases[i]
    return act


weights = Array([[[0 for k in range(10)] for j in range(10)] for i in range(2)])
biases = Array([[0 for j in range(10)] for i in range(2)])
input = Array([0] * 10)
fn(weights, biases, input)
