using Pkg
Pkg.add("DataFrames")
using DataFrames
using Statistics
using Random

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

    X = Matrix{Float64}(undef, n, 128)
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
    count(ismissing, data[!, col]) for col in feature_cols]

missing_total = sum(missing_by_feature)
missing_percentage = 100 * missing_total / (nrow(data) * length(feature_cols))

missing_by_batch = [
    sum(count(ismissing, data[data.batch .== batch, col]) for col in feature_cols)
    for batch in sort(unique(data.batch))]

println("\n========== VALORES AUSENTES ==========")
println("Porcentaje global: ", missing_percentage, "%")
println("Porcentaje por variable: ", all(iszero, missing_by_feature) ? "0% en las 128 variables" : 100 .* missing_by_feature ./ nrow(data))
println("Valores ausentes por batch: ", missing_by_batch)

 

