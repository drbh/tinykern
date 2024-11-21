# tinykern

this repo is a small exploration into writing `torch` extension kernels with `uv` tooling. 

the goal is to provide a simple example of how to write a `torch` extension kernel in `C++` and compile it with `uv` tooling and provide an example for more meaningful use cases. 

speed of iteration is everything - hopefully this helps speed up kernel development for some folks.

## Build and Run

we want to build wheels, this command will build and subsequently install the wheel in the virtual environment

```bash
make build
```

now we can run the exmaple which will load the extension via the wheel and run the matmul operation

```bash
make run-example
```

## Development

additionally, we can avoid the wheel building process and run the example directly. in this method we'll use `torch.utils.cpp_extension.load_inline` to compile the extension on the fly. this does not produce a wheel, but is helpful for reducing iteration time.

```bash
make jit-run
```