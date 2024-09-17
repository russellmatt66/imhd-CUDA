#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/io/csv.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/device/default_memory_resource.hpp>
#include <iostream>

/* EXAMPLE CODE USING RAPIDS C++ bindings to write device data to csv */

int main() {
    // Example device pointers and sizes (replace with actual device pointers)
    float* d_float_data;
    int* d_int_data_1;
    int* d_int_data_2;
    int* d_int_data_3;
    size_t num_rows = 1000;

    // Allocate and initialize device memory (for demonstration purposes)
    cudaMalloc(&d_float_data, num_rows * sizeof(float));
    cudaMalloc(&d_int_data_1, num_rows * sizeof(int));
    cudaMalloc(&d_int_data_2, num_rows * sizeof(int));
    cudaMalloc(&d_int_data_3, num_rows * sizeof(int));
    // Initialize data here...

    // Create cuDF columns from device pointers
    auto float_column = cudf::make_numeric_column(cudf::data_type(cudf::type_id::FLOAT32), num_rows);
    auto int_column_1 = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), num_rows);
    auto int_column_2 = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), num_rows);
    auto int_column_3 = cudf::make_numeric_column(cudf::data_type(cudf::type_id::INT32), num_rows);

    // Copy data from device pointers to cuDF columns
    cudaMemcpy(float_column->mutable_view().data<float>(), d_float_data, num_rows * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(int_column_1->mutable_view().data<int>(), d_int_data_1, num_rows * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(int_column_2->mutable_view().data<int>(), d_int_data_2, num_rows * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(int_column_3->mutable_view().data<int>(), d_int_data_3, num_rows * sizeof(int), cudaMemcpyDeviceToDevice);

    // Create a table from the columns
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(float_column));
    columns.push_back(std::move(int_column_1));
    columns.push_back(std::move(int_column_2));
    columns.push_back(std::move(int_column_3));

    auto table = std::make_unique<cudf::table>(std::move(columns));

    // Define the output file path
    std::string output_file_path = "output.csv";

    // Set up CSV writer options
    cudf::io::csv_writer_options options = cudf::io::csv_writer_options::builder(cudf::io::sink_info{output_file_path}, table->view());

    // Write the table to the CSV file
    cudf::io::write_csv(options);

    std::cout << "Table saved to " << output_file_path << std::endl;

    // Clean up device memory
    cudaFree(d_float_data);
    cudaFree(d_int_data_1);
    cudaFree(d_int_data_2);
    cudaFree(d_int_data_3);

    return 0;
}
