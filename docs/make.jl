using Documenter
using OpticalFibers

makedocs(
    sitename = "OpticalFibers",
    format = Documenter.HTML(),
    modules = [OpticalFibers]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/DanielHolleufer/OpticalFibers.jl",
    devbranch = "main"
)
