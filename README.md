# Steiner

Compute all conics tangent to five given conics.

## Installation

This requires `Julia 1.5`.
```sh
git clone https://github.com/JuliaHomotopyContinuation/Steiner.jl.git steiner
cd steiner
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```


## Start service

```sh
julia --project=. -e 'using Steiner; start_server()';
```

Then you should be able to go to [http://localhost:3264](http://localhost:3264)
and compute your conics.
