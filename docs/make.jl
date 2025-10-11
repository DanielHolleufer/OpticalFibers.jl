using Documenter
using OpticalFibers
using Revise

Revise.revise()

DocMeta.setdocmeta!(OpticalFibers, :DocTestSetup, :(using OpticalFibers); recursive=true)

makedocs(;
    sitename="OpticalFibers.jl",
    format=Documenter.HTML(),
    modules=[OpticalFibers],
    authors="Daniel Holleufer <daniel.holleufer@gmail.com>",
    pages=[
        "Home" => "index.md",
    ]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(;
    repo="github.com/DanielHolleufer/OpticalFibers.jl",
    devbranch="main"
)
