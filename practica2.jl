using Pkg
Pkg.add("DataFrames")
using DataFrames

using Statistics
using Random
using Pkg

Pkg.add("MLJ")
Pkg.add("MLJModelInterface")
using MLJ
import MLJModelInterface

# 1. PREPARACIÓN DE DATOS
## 1.1 Cargar el conjunto de datos

# directorio donde hemos descomprimido los datos
data_dir = joinpath(@__DIR__, "gas+sensor+array+drift+dataset+at+different+concentrations")

if !isdir(data_dir)
    error("No se encuentra la carpeta de los datos: $data_dir")
end

# relacion del codigo numerico con cada gas
const GAS_LABELS = Dict(
    1 => "Ethanol",
    2 => "Ethylene",
    3 => "Ammonia",
    4 => "Acetaldehyde",
    5 => "Acetone",
    6 => "Toluene"
)

## funcion para cargar un batch de datos

function loadGasBatch(filepath, batch_id)
    lines = readlines(filepath)
    n = length(lines)

    X = fill(NaN, n, 128)
    concentrations = Vector{Float64}(undef, n)
    gases = Vector{String}(undef, n)

    for (i, line) in enumerate(lines)
        tokens = split(line)
        metadata = split(tokens[1], ';')

        gases[i] = GAS_LABELS[parse(Int, metadata[1])]
        concentrations[i] = parse(Float64, metadata[2])

        for token in tokens[2:end]
            feature = split(token, ':')
            X[i, parse(Int, feature[1])] = parse(Float64, feature[2])
        end
    end

    #creamos una tabla para almacenar todos los datos 
    df = DataFrame(X, [Symbol("sensor_$j") for j in 1:128])
    insertcols!(df, 1,
        :batch => fill(batch_id, n),
        :concentration_ppmv => concentrations,
        :gas => gases
    )

    return df
end



## 1.2 Unificar todos los batches

batch_files = ["batch$j.dat" for j in 1:10]

batch_datasets = [
    loadGasBatch(joinpath(data_dir, file), j)
    for (j, file) in enumerate(batch_files)]

data = vcat(batch_datasets...) ## unimos todos los batches en un solo dataset

println("Dataset completo: ", size(data))

# columnas de las 128 caracteristicas
feature_cols = [Symbol("sensor_$j") for j in 1:128]


## 1.3 Descripción del dataset

println("\n========== DESCRIPCION DEL DATASET ==========")
println("Instancias: ", nrow(data))
println("Caracteristicas: ", length(feature_cols))
println("Clases: ", length(unique(data.gas)))
println("Batches: ", length(unique(data.batch)))

println("\nDistribucion de clases:")
for gas in sort(unique(data.gas))
    println(gas, ": ", count(==(gas), data.gas))
end

println("\nDistribucion por batch:")
for batch in sort(unique(data.batch))
    println("Batch ", batch, ": ", count(==(batch), data.batch))
end


## 1.4 Análisis de valores faltantes

missing_by_feature = [
    count(isnan, data[!, col]) for col in feature_cols]

missing_total = sum(missing_by_feature)
missing_percentage = 100 * missing_total / (nrow(data) * length(feature_cols))

missing_by_batch = [
    sum(count(isnan, data[data.batch .== batch, col]) for col in feature_cols)
    for batch in sort(unique(data.batch))]

println("\n========== VALORES AUSENTES ==========")
println("Porcentaje global: ", missing_percentage, "%")
println("Porcentaje por variable: ", all(iszero, missing_by_feature) ? "0% en las 128 variables" : 100 .* missing_by_feature ./ nrow(data))
println("Valores ausentes por batch: ", missing_by_batch)

## 1.5 Partición temporal hold-out

# ultimo batch para test (aunque supone un 20% del total de instancias)
test_batch = maximum(data.batch)

train_data = data[data.batch .< test_batch, :]
test_data = data[data.batch .== test_batch, :]

println("\n========== PARTICION HOLDOUT ==========")

println("Batches de entrenamiento: ", sort(unique(train_data.batch)))
println("Batches de test: ", unique(test_data.batch))

println("Instancias de entrenamiento: ", nrow(train_data))
println("Instancias de test: ", nrow(test_data))

println("Porcentaje de test: ",
    round(100 * nrow(test_data) / nrow(data), digits=2), "%")


# 1.6 Validación cruzada 5-fold
cv_batches = sort(unique(train_data.batch))

folds = []

for i in 1:5

    # batches de entrenamiento
    train_batches_fold = cv_batches[1:i+3]

    # siguiente batch para validacion
    valid_batch = cv_batches[i+4]

    # separar las observaciones
    train_fold = train_data[
        [b in train_batches_fold for b in train_data.batch], :]

    valid_fold = train_data[
        train_data.batch .== valid_batch, :]

    # guardar la particion
    push!(
        folds,
        (train = train_fold, validation = valid_fold))
end

# Mostrar resultados
println("\n========== VALIDACION 5-FOLD ==========")

for (i, fold) in enumerate(folds)

    println("\nFold ", i)
    println("Batches entrenamiento: ", sort(unique(fold.train.batch)))
    println("Batch validacion: ", unique(fold.validation.batch))
    println("Instancias entrenamiento: ", nrow(fold.train))
    println("Instancias validacion: ", nrow(fold.validation))
end

## 1.7 Normalización de características

# nodo de normalizacion compatible con MLJ
mutable struct MinMaxNode <: MLJModelInterface.Unsupervised
end

# tipo de datos que recibe y devuelve
MLJModelInterface.input_scitype(::Type{<:MinMaxNode}) =
    MLJModelInterface.Table(MLJModelInterface.Continuous)

MLJModelInterface.output_scitype(::Type{<:MinMaxNode}) =
    MLJModelInterface.Table(MLJModelInterface.Continuous)


# aprender los minimos y maximos del entrenamiento
function MLJModelInterface.fit(::MinMaxNode, verbosity::Int, X)

    df = DataFrame(X)
    columnas = Symbol.(names(df))
    minimos = [minimum(df[!, c]) for c in columnas]
    maximos = [maximum(df[!, c]) for c in columnas]

    return (columnas, minimos, maximos), nothing, NamedTuple()
end


# aplicar la normalizacion utilizando los valores aprendidos
function MLJModelInterface.transform(::MinMaxNode, fitresult, Xnew)

    columnas, minimos, maximos = fitresult

    df = DataFrame(Xnew)

    for j in eachindex(columnas)
        rango = maximos[j] - minimos[j]
        df[!, columnas[j]] =
            (df[!, columnas[j]] .- minimos[j]) ./ (rango == 0 ? 1 : rango)
    end

    return df
end

# funcion para incorporar Min-Max a cualquier clasificador
minmax_pipeline(model) = MinMaxNode() |> model
