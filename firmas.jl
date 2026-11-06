# Grupo 1.1 B (Viernes): Noel Allegue Vázquez, Ander Asteinza García, Jairo Candel Fariña, Mateo Lourido Pena

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 1 --------------------------------------------
# ----------------------------------------------------------------------------------------------

import FileIO.load
using DelimitedFiles

function fileNamesFolder(folderName::String, extension::String)
    extension = uppercase(extension);
    fileNames = filter(f -> endswith(uppercase(f), ".$extension"), readdir(folderName));
    fileNames = first.(splitext.(fileNames))
    return fileNames
end;



function loadDataset(datasetName::String, datasetFolder::String;
    datasetType::DataType=Float64)
    fileName = datasetName * ".tsv"
    filePath = abspath(joinpath(datasetFolder, fileName))
    data = readdlm(filePath, '\t')
    data = readdlm(filePath, '\t')

    # columnas
    headers = data[1, :]
    # vector con los índices de la columna de salida
    target_col = findall(headers .== "target")[1] 
    
    #vector con los índices de columnas menos la de target
    input_cols = setdiff(1:size(data, 2), target_col) 

    # conversión de tipo sin bucles
    inputs = datasetType.(data[2:end, input_cols])
    
    # salidas sin convertir
    raw_targets = data[2:end, target_col]

    # unique obtiene los valores únicos y setdiff los compara con [0, 1]. 

    if isempty(setdiff(unique(raw_targets), [0, 1])) #clasificación binaria
        targets = Bool.(raw_targets)
    else
        targets = datasetType.(raw_targets) #float64 para regresión
    end

    return (inputs, targets)

end;



function datasetsDescription(datasetFolder::String, classification::Bool)

    names = fileNamesFolder(datasetFolder, "tsv")
    datasets = loadDataset.(names, Ref(datasetFolder))

    # tipo del vector de targets de cada dataset
    target_types = eltype.(last.(datasets))

    # máscara booleana para filtrar. 
    mask = classification ? (target_types .== Bool) : (target_types .!= Bool)
    
    valid_names = names[mask]
    valid_datasets = datasets[mask]
     
    inputs = first.(valid_datasets) 
    num_instances = size.(inputs, 1)
    num_attributes = size.(inputs, 2)
    
    sizes = num_instances .* num_attributes
    sort_perm = sortperm(sizes)
    
    return hcat(valid_names[sort_perm], num_instances[sort_perm], num_attributes[sort_perm])

end;



function intervalDiscreteVector(data::AbstractArray{<:Real,1})
    
    uniqueData = sort(unique(data));
    #diferencias entre elementos consecutivos
    differences = sort(diff(uniqueData));
    #tomamos la menor diferencia
    minDifference = differences[1];

    isInteger(x::Float64, tol::Float64) = abs(round(x)-x) < tol
    return all(isInteger.(differences./minDifference, 1e-3)) ? minDifference : 0

end;


function cyclicalEncoding(data::AbstractArray{<:Real,1})
    m = intervalDiscreteVector(data)
    minData, maxData = extrema(data)

    angulos = 2π .* (data .- minData) ./ (maxData - minData + m)
    senos = sin.(angulos)
    cosenos = cos.(angulos)
    return (senos, cosenos)

end;



function loadStreamLearningDataset(datasetFolder::String; datasetType::DataType=Float64)
    
    data = readdlm(abspath(joinpath(datasetFolder, "elec2_data.dat")))
    labels = readdlm(abspath(joinpath(datasetFolder, "elec2_label.dat")))
    # salidas a vector de booleanos
    targets = vec(Bool.(labels))
    # eliminar las columnas date y nswprice
    columnsFinal = setdiff(1:size(data, 2), [1, 4])
    processedData = data[:, columnsFinal]
    # ahora la primera columna es day
    day = processedData[:, 1]

    daySenos, dayCosenos = cyclicalEncoding(day)

    # eliminamos la columna day original
    inputsFinal = processedData[:, 2:end]

    inputs = datasetType.(hcat(daySenos, dayCosenos, inputsFinal))
    return (inputs, targets)
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 2 --------------------------------------------
# ----------------------------------------------------------------------------------------------

using MLJ, LIBSVM, MLJLIBSVMInterface
SVMClassifier = MLJ.@load SVC pkg=LIBSVM verbosity=0
predict(model::Machine{MLJLIBSVMInterface.SVC, MLJLIBSVMInterface.SVC, true}, inputs::AbstractArray) = (outputs = MLJ.predict(model, MLJ.table(inputs)); return convert(Vector, levels(outputs)[int(outputs)]); )



using Base.Iterators
using StatsBase

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function batchInputs(batch::Batch)
    return batch[1]
end;

function batchTargets(batch::Batch)
    return batch[2]
end;

function batchLength(batch::Batch)
    # Devuelve el número de elementos del vector de salidas deseadas
    return length(batchTargets(batch))
end;

function selectInstances(batch::Batch, indices::Any)
    inputs = batchInputs(batch)
    targets = batchTargets(batch)

    inputsSelec = inputs[indices, :]
    targetsSelec = targets[indices]

    return (inputsSelec, targetsSelec)
end;

function joinBatches(batch1::Batch, batch2::Batch)
    inputs = vcat(batchInputs(batch1), batchInputs(batch2))
    targets = vcat(batchTargets(batch1), batchTargets(batch2))
 
    return (inputs, targets)
end;

using Random
function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    # índices de todas las instancias
    indices = 1:batchLength(dataset)

    # si shuffleRows es true desordenamos los índices 
    indices = shuffleRows ? shuffle(indices) : indices

    batches = [selectInstances(dataset, collect(batchIndices)) for batchIndices in Base.Iterators.partition(indices, batchSize)]

    return batches
end;

function trainSVM(dataset::Batch, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.,
    supportVectors::Batch=( Array{eltype(dataset[1]),2}(undef,0,size(dataset[1],2)) , Array{eltype(dataset[2]),1}(undef,0) ) )
    
    joined_dataset = joinBatches(supportVectors, dataset)
    model = SVMClassifier(
         kernel =
            kernel=="linear" ? LIBSVM.Kernel.Linear :
            kernel=="rbf" ? LIBSVM.Kernel.RadialBasis :
            kernel=="poly" ? LIBSVM.Kernel.Polynomial :
            kernel=="sigmoid" ? LIBSVM.Kernel.Sigmoid : nothing, 
        cost = Float64(C),
        gamma = Float64(gamma),
        degree = Int32( degree),
    coef0 = Float64(coef0)); 
       
    mach = machine(model, 
        MLJ.table(batchInputs(joined_dataset)), 
        categorical(batchTargets(joined_dataset); levels=[false, true]))
    MLJ.fit!(mach)
    indicesNewSupportVectors = sort(mach.fitresult[1].SVs.indices)
    N = batchLength(supportVectors)
    idx_sv = indicesNewSupportVectors[indicesNewSupportVectors .<= N]
    idx_dataset = indicesNewSupportVectors[indicesNewSupportVectors .> N] .- N
    new_sv = joinBatches(selectInstances(supportVectors, idx_sv), selectInstances(dataset, idx_dataset))
    
    return (mach, new_sv, (idx_sv, idx_dataset))
end;

function trainSVM(batches::AbstractArray{<:Batch,1}, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    sv = (Array{eltype(batches[1][1]),2}(undef,0,size(batches[1][1],2)), Array{eltype(batches[1][2]),1}(undef,0))
    local mach
    
    for batch in batches
        mach, sv, _ = trainSVM(batch, kernel, C; degree=degree, gamma=gamma, coef0=coef0, supportVectors=sv)
    end
    
    return mach
end;


    

# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 3 --------------------------------------------
# ----------------------------------------------------------------------------------------------


function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)
    # carga de datos completos
    full_dataset = loadStreamLearningDataset(datasetFolder)
    
    # tomar los primeros elementos como memoria
    memory_indices = 1:windowSize
    memory = selectInstances(full_dataset, collect(memory_indices))
    
    # con el resto de datos sacamos el flujo (stream)
    stream_indices = (windowSize + 1):batchLength(full_dataset)
    stream_dataset = selectInstances(full_dataset, collect(stream_indices))
    
    # dividir el flujo en batches sin desordenar (shuffleRows = false)
    batches = divideBatches(stream_dataset, batchSize; shuffleRows = false)
    
    return (memory, batches)
end;


function addBatch!(memory::Batch, newBatch::Batch)
    num_new = batchLength(newBatch)
    mem_len = batchLength(memory)
    
    inputs_mem = batchInputs(memory)
    targets_mem = batchTargets(memory)
    
    inputs_new = batchInputs(newBatch)
    targets_new = batchTargets(newBatch)
    
    # desplazar los datos antiguos para arriba (descartando los primeros)
    inputs_mem[1:(mem_len - num_new), :] .= inputs_mem[(num_new + 1):mem_len, :]
    targets_mem[1:(mem_len - num_new)] .= targets_mem[(num_new + 1):mem_len]
    
    # copiar el nuevo lote al final de la memoria
    inputs_mem[(mem_len - num_new + 1):end, :] .= inputs_new
    targets_mem[(mem_len - num_new + 1):end] .= targets_new
    
    return nothing
end;


function streamLearning_SVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    # memoria y flujo
    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    
    # entrenar modelo inicial con la memoria
    mach, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    
    num_batches = length(batches)
    accuracies = Vector{Float64}(undef, num_batches)
    
    for i in 1:num_batches
        current_batch = batches[i]
        
        # evaluar en el batch actual
        preds = predict(mach, batchInputs(current_batch))
        accuracies[i] = mean(preds .== batchTargets(current_batch))
        
        # actualizamos memoria y reentrenamos
        addBatch!(memory, current_batch)
        mach, _, _ = trainSVM(memory, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    end
    
    return accuracies
end;


function streamLearning_ISVM(datasetFolder::String, windowSize::Int, batchSize::Int, kernel::String, C::Real;
    degree::Real=1, gamma::Real=2, coef0::Real=0.)
    
    # inicializamos con la funcion creada
    memory_warm, batches = initializeStreamLearningData(datasetFolder, batchSize, batchSize)
    
    # entrenamiento inicial
    mach, current_sv, (_, sv_indices_in_warm) = trainSVM(memory_warm, kernel, C; degree=degree, gamma=gamma, coef0=coef0)
    
    # edades de los vectores soporte iniciales
    initial_ages = collect(batchSize:-1:1)
    sv_ages = initial_ages[sv_indices_in_warm]
    
    num_batches = length(batches)
    accuracies = Vector{Float64}(undef, num_batches)
    
    for i in 1:num_batches
        current_batch = batches[i]
        batch_len = batchLength(current_batch)
        
        # test sobre el nuevo batch
        preds = predict(mach, batchInputs(current_batch))
        accuracies[i] = mean(preds .== batchTargets(current_batch))
        
        # actualización de edades (sumando el tamaño del batch que acaba de llegar)
        sv_ages .+= batch_len
        
        valid_indices = findall(sv_ages .<= windowSize)
        current_sv = selectInstances(current_sv, valid_indices)
        sv_ages = sv_ages[valid_indices]
        
        # apartado de programación defensiva para detectar errores
        @assert length(sv_ages) == batchLength(current_sv) "Inconsistencia entre SV y edades"
        
        # reentrenar usando el batch actual y los SV vigentes
        mach, _, (idx_old_sv, idx_new_batch) = trainSVM(current_batch, kernel, C; 
                                                        degree=degree, gamma=gamma, coef0=coef0, 
                                                        supportVectors=current_sv)
        
        # nuevos vectores soporte
        sv_from_old = selectInstances(current_sv, idx_old_sv)
        sv_from_new = selectInstances(current_batch, idx_new_batch)
        current_sv = joinBatches(sv_from_old, sv_from_new)
        
        # actualizamos las edades combinando ambos grupos
        ages_from_old = sv_ages[idx_old_sv]
        ages_for_new_batch = collect(batch_len:-1:1)[idx_new_batch]
        sv_ages = vcat(ages_from_old, ages_for_new_batch)
    end
    
    return accuracies
end;


function euclideanDistances(dataset::Batch, instance::AbstractArray{<:Real,1})
    inputs = batchInputs(dataset)
    sq_diff = (inputs .- instance') .^ 2
    return vec(sqrt.(sum(sq_diff, dims=2)))
end;


function nearestElements(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    dists = euclideanDistances(dataset, instance)
    closest_indices = partialsortperm(dists, 1:k)
    return selectInstances(dataset, closest_indices)
end;


function predictKNN(dataset::Batch, instance::AbstractArray{<:Real,1}, k::Int)
    closest_batch = nearestElements(dataset, instance, k)
    targets = batchTargets(closest_batch)
    # devolver el valor mayoritario (la moda) de los vecinos
    return mode(targets)
end;


function predictKNN(dataset::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    return [predictKNN(dataset, Vector(row), k) for row in eachrow(instances)]
end;


function streamLearning_KNN(datasetFolder::String, windowSize::Int, batchSize::Int, k::Int)
    
    memory, batches = initializeStreamLearningData(datasetFolder, windowSize, batchSize)
    num_batches = length(batches)
    accuracies = Vector{Float64}(undef, num_batches)
    
    for i in 1:num_batches
        current_batch = batches[i]
        
        preds = predictKNN(memory, batchInputs(current_batch), k)
        accuracies[i] = mean(preds .== batchTargets(current_batch))
        
        addBatch!(memory, current_batch)
    end
    
    return accuracies
end;




# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 4 --------------------------------------------
# ----------------------------------------------------------------------------------------------


abstract type ModelTree end

mutable struct ModelTreeNode <: ModelTree
    feature    :: Int
    threshold  :: Real
    model      :: Any
    leftChild  :: ModelTree
    rightChild :: ModelTree
end

struct ModelTreeLeaf <: ModelTree
    model      :: Any
    n          :: Int
end


function mse(outputs::Union{Real,AbstractVector{<:Real}}, targets::AbstractVector{<:Real})
    # broadcast para que funcione tanto si es un único valor o un vector
    return mean((outputs .- targets) .^ 2)
end;
  
function candidateThresholds(x::AbstractVector{<:Real})
    valores = sort(unique(x))
    return (valores[1:end-1] .+ valores[2:end]) ./ 2
end;

#esta funcion no estaba en el firmas proporcionado pero el pdf la pide
function splitDataset(inputs::AbstractMatrix{<:Real}, numFeature::Int, threshold::Real)
    mask = inputs[:, numFeature] .<= threshold
    return (inputs[mask, :], inputs[.!mask, :], mask)
end

function splitDataset(dataset::Batch, numFeature::Int, threshold::Real)
    # obtenemos la máscara con la función anterior y seleccionamos las instancias
    datoMenorThreshold, datoMayorThreshold, mask = splitDataset(batchInputs(dataset), numFeature, threshold)
    return (selectInstances(dataset, mask), selectInstances(dataset, .!mask), mask)
end


function calculateErrorSplit(dataset::Batch, numFeature::Int, threshold::Real)

    #la parte1 seria la rama izquierda y la parte2 la rama derecha del arbol
    parte1, parte2, _ = splitDataset(dataset, numFeature, threshold) 
    targets_parte1 = batchTargets(parte1)
    targets_parte2 = batchTargets(parte2)

    error_parte1 = length(targets_parte1) * mse(mean(targets_parte1), targets_parte1)
    error_parte2 = length(targets_parte2) * mse(mean(targets_parte2), targets_parte2)
    return error_parte1 + error_parte2
end;

function bestSplit(dataset::Batch)
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)
    n_instancias = batchLength(dataset)
    T = eltype(inputs)

    
    if n_instancias <= 1
        return (-1, T(NaN))
    end

    # error del conjunto sin dividir
    bestError = n_instancias * mse(mean(targets), targets)
    bestFeature = -1
    bestThreshold = T(NaN)

    for caracteristica in 1:size(inputs, 2)
        for umbral in candidateThresholds(inputs[:, caracteristica])
            error = calculateErrorSplit(dataset, caracteristica, umbral)
            if error < bestError
                bestError = error
                bestFeature = caracteristica
                bestThreshold = umbral
            end
        end
    end

    return (bestFeature, bestThreshold)
end;


function buildRegressionTree(dataset::Batch;
    max_depth        :: Int = -1,
    min_samples_leaf :: Int =  1
)
    n_instancias = batchLength(dataset)
    model = mean(batchTargets(dataset))

    # caso base recursivo -- pocas muestras o se llega a la maxima profundidad
    if n_instancias <= min_samples_leaf || max_depth == 0
        return ModelTreeLeaf(model, n_instancias)
    end

    caracteristica, umbral = bestSplit(dataset)

    # ninguna división mejora el error actual
    if caracteristica == -1
        return ModelTreeLeaf(model, n_instancias)
    end

    parte1, parte2, _ = splitDataset(dataset, caracteristica, umbral)
    parte1Tree  = buildRegressionTree(parte1;  max_depth = max_depth - 1, min_samples_leaf = min_samples_leaf)
    parte2Tree = buildRegressionTree(parte2; max_depth = max_depth - 1, min_samples_leaf = min_samples_leaf)

    return ModelTreeNode(caracteristica, umbral, model, parte1Tree, parte2Tree)
end;

function predict(model::Real, inputs::AbstractMatrix{<:Real})
    return fill(model, size(inputs, 1))
end;

function predict(tree::ModelTreeLeaf, inputs::AbstractMatrix{<:Real})
    return predict(tree.model, inputs)
end;

function predict(tree::ModelTreeNode, inputs::AbstractMatrix{<:Real})
    parte1, parte2, mask = splitDataset(inputs, tree.feature, tree.threshold)

    # predicciones de cada hijo 
    outputsparte1  = predict(tree.leftChild, parte1)
    outputsparte2 = predict(tree.rightChild, parte2)

    #ensamblar resultados en un vector de salida del mismo tipo que los outputs de cada rama
    outputs = Vector{promote_type(eltype(outputsparte1), eltype(outputsparte2))}(undef, size(inputs, 1))
    outputs[mask]   = outputsparte1
    outputs[.!mask] = outputsparte2
    return outputs
end;



# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 5 --------------------------------------------
# ----------------------------------------------------------------------------------------------


EpsilonSVR = MLJ.@load EpsilonSVR pkg=LIBSVM verbosity=0
predict(model::Machine{MLJLIBSVMInterface.EpsilonSVR, MLJLIBSVMInterface.EpsilonSVR, true}, inputs::AbstractArray) = MLJ.predict(model, inputs)

using SymDoME
function predict(model::SymDoME.Tree, inputs::AbstractMatrix{<:Real})
    outputs = evaluateTree(model, inputs)

    return outputs isa Real ? fill(outputs, size(inputs, 1)) : vec(outputs)
end;

LinearModel = Tuple{Real, AbstractArray{<:Real,1}}
function predict(model::LinearModel, inputs::AbstractMatrix{<:Real})

    return inputs * model[2] .+ model[1]

end;


function trainRegressionModel(configuration::Dict, dataset::Batch)
    
    return trainRegressionModel(Val(configuration["modelType"]), configuration, dataset)

end;

function trainRegressionModel(::Val{:constant}, configuration::Dict, dataset::Batch)

    targets = batchTargets(dataset)
    return eltype(targets)(mean(targets))

end;

function trainRegressionModel(::Val{:linear}, configuration::Dict, dataset::Batch)
    
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)

    # Crear una columna de unos del mismo tipo que las entradas
    onesColumn = ones(eltype(inputs), size(inputs, 1))

    extendedInputs = hcat(onesColumn, inputs)

    coefficients = extendedInputs \ targets

    intercept = coefficients[1]
    regressionCoefficients = coefficients[2:end]

    return (intercept, regressionCoefficients)::LinearModel

end;

function trainRegressionModel(::Val{:DoME}, configuration::Dict, dataset::Batch)

    result = dome(batchInputs(dataset), batchTargets(dataset); maximumNodes=configuration["maximumNodes"])

    return first(result)

end;

function trainRegressionModel(::Val{:SVR}, configuration::Dict, dataset::Batch)
    
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)

    # Leer el kernel indicado o rbf por defecto
    kernelName = lowercase(String(get(configuration, "kernel", "rbf")))

    # Convertir el nombre del kernel al valor utilizado por LIBSVM
    if kernelName == "linear"
        kernel = LIBSVM.Kernel.Linear
    elseif kernelName == "poly"
        kernel = LIBSVM.Kernel.Polynomial
    elseif kernelName == "sigmoid"
        kernel = LIBSVM.Kernel.Sigmoid
    else
        kernel = LIBSVM.Kernel.RadialBasis
    end

    model = EpsilonSVR(
        kernel  = kernel,
        epsilon = Float64(get(configuration, "epsilon", 0.1)),
        cost    = Float64(get(configuration, "C", 1.0)),
        gamma   = Float64(
            get(configuration, "gamma", 1 / size(inputs, 2))
        ),
        degree  = Int32(get(configuration, "degree", 3)),
        coef0   = Float64(get(configuration, "coef0", 0.0))
    )

    mach = machine(model, MLJ.table(inputs), targets)

    MLJ.fit!(mach, verbosity=0)

    return mach

end;



function calculateErrorSplit(dataset::Batch, numFeature::Int, threshold::Real, modelOutputs::AbstractVector{<:Real})
    
    if length(unique(modelOutputs)) == 1
        return calculateErrorSplit(dataset, numFeature, threshold)
    end

    parte1, parte2, indicesParte1 = splitDataset(dataset, numFeature, threshold)

    indicesParte2 = .!indicesParte1

    targetsParte1 = batchTargets(parte1)
    targetsParte2 = batchTargets(parte2)

    outputsParte1 = modelOutputs[indicesParte1]
    outputsParte2 = modelOutputs[indicesParte2]
    
    errorParte1 = length(targetsParte1) * mse(outputsParte1, targetsParte1)
    errorParte2 = length(targetsParte2) * mse(outputsParte2, targetsParte2)

    return errorParte1 + errorParte2

end;



function bestSplit(dataset::Batch, modelOutputs::AbstractVector{<:Real})
    
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)
    n_instancias = batchLength(dataset)
    T = eltype(inputs)

    
    if n_instancias <= 1
        return (-1, T(NaN))
    end

    bestError = n_instancias * mse(modelOutputs, targets)

    bestFeature = -1
    bestThreshold = T(NaN)

    for caracteristica in 1:size(inputs, 2)

        for umbral in candidateThresholds(inputs[:, caracteristica])

            error = calculateErrorSplit(dataset, caracteristica, umbral, modelOutputs)

            if error < bestError
                bestError = error
                bestFeature = caracteristica
                bestThreshold = T(umbral)
            end
        end
    end

    return (bestFeature, bestThreshold)

end;

function buildModelTree(configuration::Dict, dataset::Batch;
    max_depth        :: Int = -1,
    min_samples_leaf :: Int =  1
)

    n_instancias = batchLength(dataset)

    model = trainRegressionModel(configuration, dataset)

    if n_instancias <= min_samples_leaf || max_depth == 0
        return ModelTreeLeaf(model, n_instancias)
    end

    modelOutputs = predict(model, batchInputs(dataset))

    caracteristica, umbral = bestSplit(dataset, modelOutputs)

    if caracteristica == -1
        return ModelTreeLeaf(model, n_instancias)
    end

    parte1, parte2, _ = splitDataset(dataset, caracteristica, umbral)

    parte1Tree = buildModelTree(configuration, parte1; max_depth=max_depth - 1, min_samples_leaf=min_samples_leaf)
    parte2Tree = buildModelTree(configuration, parte2; max_depth=max_depth - 1, min_samples_leaf=min_samples_leaf)

    return ModelTreeNode(caracteristica, umbral, model, parte1Tree, parte2Tree)

end;


function trainRegressionModel(::Val{:ModelTree}, configuration::Dict, dataset::Batch)

    max_depth = get(configuration, "max_depth", -1)
    min_samples_leaf = get(configuration, "min_samples_leaf", 1)
    localConfig = configuration["localModelConfiguration"]

    return buildModelTree(localConfig, dataset; max_depth=max_depth, min_samples_leaf=min_samples_leaf)

end;


function numInstances(tree::ModelTreeLeaf)

    return tree.n

end;


function numInstances(tree::ModelTreeNode)

    return numInstances(tree.leftChild) + numInstances(tree.rightChild)

end;


function predict(tree::ModelTreeLeaf, inputs::AbstractMatrix{<:Real}, k::Int)

    return predict(tree, inputs)

end;


function predict(tree::ModelTreeNode, inputs::AbstractMatrix{<:Real}, k::Int)

    if k <= 0
        return predict(tree, inputs)
    end

    parte1, parte2, mask = splitDataset(inputs, tree.feature, tree.threshold)

    p1 = predict(tree.leftChild, parte1, k)
    p2 = predict(tree.rightChild, parte2, k)
    q  = predict(tree.model, inputs)

    n1 = numInstances(tree.leftChild)
    n2 = numInstances(tree.rightChild)

    T = promote_type(eltype(p1), eltype(p2), eltype(q))
    outputs = Vector{T}(undef, size(inputs, 1))
    outputs[mask]   = (n1 .* p1 .+ k .* q[mask]) ./ (n1 + k)
    outputs[.!mask] = (n2 .* p2 .+ k .* q[.!mask]) ./ (n2 + k)

    return outputs

end;


function prune!(tree::ModelTreeLeaf, validationDataset::Batch)

    return tree

end;

function prune!(tree::ModelTreeNode, validationDataset::Batch)

    if batchLength(validationDataset) == 0
        return tree
    end

    valLeft, valRight, _ = splitDataset(validationDataset, tree.feature, tree.threshold)

    tree.leftChild  = prune!(tree.leftChild, valLeft)
    tree.rightChild = prune!(tree.rightChild, valRight)

    valInputs  = batchInputs(validationDataset)
    valTargets = batchTargets(validationDataset)

    errorNode  = mse(predict(tree, valInputs), valTargets)
    errorModel = mse(predict(tree.model, valInputs), valTargets)

    if errorModel <= errorNode
        return ModelTreeLeaf(tree.model, numInstances(tree))
    end

    return tree
    
end;





# ----------------------------------------------------------------------------------------------
# ------------------------------------- Ejercicio 6 --------------------------------------------
# ----------------------------------------------------------------------------------------------


struct KNN
    dataset                 :: Batch
    k                       :: Int
    globalModel             :: Any # isnothing(globalModel) -> only local predictions (alpha is not used)
    localModelConfiguration :: Dict
    alpha                   :: AbstractFloat # isnan(alpha) -> residual correction
end


function trainRegressionModel(::Val{:KNN}, configuration::Dict, dataset::Batch)
    k = get(configuration, "k", 5)

    globalModel = haskey(configuration, "globalModelConfiguration") ?
        trainRegressionModel(configuration["globalModelConfiguration"], dataset) :
        nothing

    localModelConfiguration = get(configuration, "localModelConfiguration", Dict("modelType" => :constant))

    alpha = Float64(get(configuration, "alpha", NaN))

    return KNN(dataset, k, globalModel, localModelConfiguration, alpha)
end;


function predict(model::KNN, instance::AbstractArray{<:Real,1})
    neighbors = nearestElements(model.dataset, instance, model.k)
    neighborInputs = batchInputs(neighbors)
    neighborTargets = batchTargets(neighbors)

    instanceMatrix = reshape(instance, 1, :)

    hasGlobal = !isnothing(model.globalModel)
    residualMode = hasGlobal && isnan(model.alpha)

    localTargets = residualMode ?
        neighborTargets .- predict(model.globalModel, neighborInputs) :
        neighborTargets

    localModel = trainRegressionModel(model.localModelConfiguration, (neighborInputs, localTargets))
    localOutput = first(predict(localModel, instanceMatrix))

    !hasGlobal && return localOutput

    globalOutput = first(predict(model.globalModel, instanceMatrix))

    return residualMode ?
        globalOutput + localOutput :
        model.alpha * localOutput + (1 - model.alpha) * globalOutput
end;


function predict(model::KNN, inputs::AbstractArray{<:Real,2})

    return [predict(model, Vector(row)) for row in eachrow(inputs)]
end;

