// #include <torch/torch.h>
#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> top_pool_forward(
    torch::Tensor input
) {
    // Initialize output
    torch::Tensor output = torch::zeros_like(input);

    // Get height
    int64_t width = input.size(4);

    // Copy the last column
    torch::Tensor input_temp  = input.select(4, width - 1);
    torch::Tensor output_temp = output.select(4, width - 1);
    output_temp.copy_(input_temp);

    torch::Tensor max_temp;
    for (int64_t ind = 1; ind < width; ++ind) {
        input_temp  = input.select(4, width - ind - 1);
        output_temp = output.select(4, width - ind);
        max_temp    = output.select(4, width - ind - 1);

        torch::max_out(max_temp, input_temp, output_temp);
    }

    return { 
        output
    };
}

std::vector<torch::Tensor> top_pool_backward(
    torch::Tensor input,
    torch::Tensor grad_output
) {
    auto output = torch::zeros_like(input);

    int32_t batch   = input.size(0);
    int32_t channel = input.size(1);
    int32_t depth  = input.size(2);
    int32_t height  = input.size(3);
    int32_t width   = input.size(4);

    // auto max_val = torch::zeros(torch::CUDA(torch::kFloat), {batch, channel, width});
    // auto max_ind = torch::zeros(torch::CUDA(torch::kLong),  {batch, channel, width});
    auto max_val = torch::zeros({batch, channel, depth, height}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
    auto max_ind = torch::zeros({batch, channel, depth, height}, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));

    auto input_temp = input.select(4, width - 1);
    max_val.copy_(input_temp);

    max_ind.fill_(width - 1);

    auto output_temp      = output.select(4, width - 1);
    auto grad_output_temp = grad_output.select(4, width - 1);
    output_temp.copy_(grad_output_temp);

    auto un_max_ind = max_ind.unsqueeze(4);
    // auto gt_mask    = torch::zeros(torch::CUDA(torch::kByte),  {batch, channel, width});
    // auto max_temp   = torch::zeros(torch::CUDA(torch::kFloat), {batch, channel, width});
	auto gt_mask    = torch::zeros({batch, channel, depth, height}, torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));
    auto max_temp   = torch::zeros({batch, channel, depth, height}, torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));

    for (int32_t ind = 1; ind < width; ++ind) {
        input_temp = input.select(4, width - ind - 1);
        torch::gt_out(gt_mask, input_temp, max_val);

        torch::masked_select_out(max_temp, input_temp, gt_mask);
        max_val.masked_scatter_(gt_mask, max_temp);
        max_ind.masked_fill_(gt_mask, width - ind - 1);

        grad_output_temp = grad_output.select(4, width - ind - 1).unsqueeze(4);
        output.scatter_add_(4, un_max_ind, grad_output_temp);
    }

    return {
        output
    };
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward", &top_pool_forward, "Top Pool Forward",
        py::call_guard<py::gil_scoped_release>()
    );
    m.def(
        "backward", &top_pool_backward, "Top Pool Backward",
        py::call_guard<py::gil_scoped_release>()
    );
}
