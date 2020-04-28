//
//  SimpleNeuralNetwork.swift
//  NN_Demo
//
//  Created by Cyril Garcia on 4/26/20.
//  Copyright © 2020 Cyril Garcia. All rights reserved.
//

import Cocoa
import TensorFlow
#if canImport(PythonKit)
import PythonKit
#else
import Python
#endif

struct Model {
    var weights = Tensor(shape: [3,1],scalars: [Float.random(in: 0...1),
                                                Float.random(in: 0...1),
                                                Float.random(in: 0...1)])
    
    var inputs: Tensor<Float> = Tensor(shape: [3,3], scalars: [0,0,1,1,1,1,0,1,1])
    var outputs: Tensor<Float> = Tensor(shape: [3,1], scalars: [0,1,0])
    
    func predict(_ x: Tensor<Float>) -> Tensor<Float> {
        return sigmoid(x • weights)
    }
    
    func prime(_ x: Tensor<Float>) -> Tensor<Float> {
        let vals = x.scalars.map { return $0 * (1 - $0) }
        return Tensor(shape: [3,1], scalars: vals)
    }
    
}

final class SimpleNeuralNetwork: NSViewController {
    
    var epochs = 100
    
//    var model = Model()
    
    var weights = Tensor(shape: [3,1],scalars: [Float.random(in: 0...1),
                                                Float.random(in: 0...1),
                                                Float.random(in: 0...1)])
    
    var inputs: Tensor<Float> = Tensor(shape: [3,3], scalars: [0,0,1,1,1,1,0,1,1])
    var outputs: Tensor<Float> = Tensor(shape: [3,1], scalars: [0,1,0])
    
    func predict(_ x: Tensor<Float>) -> Tensor<Float> {
        return sigmoid(x • weights)
    }
    
    func prime(_ x: Tensor<Float>) -> Tensor<Float> {
        let vals = x.scalars.map { return $0 * (1 - $0) }
        return Tensor(shape: [3,1], scalars: vals)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        let a: Tensor<Float> = [[0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]]
        
        let b: Tensor<Float> = [[-0.074010275, -0.074010275, 0.14814812],
        [ 0.12646824, 0.12646824, 0.12646824],
        [ -0.05096831, 0.14284894, 0.14284894]]
        
        print(matmul(a, b))
    }
    
    func train() {
        print(weights)
        
        for _ in 1...epochs {
            let predictedOutputs = predict(inputs)
            let error = outputs - predictedOutputs
            
            let adjustments = inputs.transposed() • error * prime(predictedOutputs)
            
            weights += adjustments
            
        }
        
        print("Weights:", weights)
        print(predict(inputs))
    }
}
