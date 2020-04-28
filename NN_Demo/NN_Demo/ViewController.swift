//
//  ViewController.swift
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

struct ColorBatch: TensorGroup {
    let features: Tensor<Float>
    let labels: Tensor<Int32>
}

let hiddenSize: Int = 10

struct ColorModel: Layer {
    
    var layer1 = Dense<Float>(inputSize: 3, outputSize: hiddenSize, activation: softmax)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3, activation: softmax)
    
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float>  {
        return input.sequenced(through: layer1, layer3)
    }
}

class ViewController: NSViewController, NSTextFieldDelegate {
    
    var epochs: Int = 500
    var learningRate: Float = 0.2
    
    var model = ColorModel()
    let locale = Python.import("locale")
    
    @IBOutlet var rField: NSTextField!
    @IBOutlet var gField: NSTextField!
    @IBOutlet var bField: NSTextField!
    @IBOutlet var labelPicker: NSPopUpButton!
    @IBOutlet var colorViewer: NSView!
    @IBOutlet var outputLabel: NSTextField!
    
    //    var trainingInputs = [Tensor<Float>]()
    //    var trainingOutputs = [Tensor<Int32>]()
    
    //    var trainingInputs: Tensor<Float> = [[0.6862745, 0.93333334, 0.58431375],
    //                                         [ 0.34901962,0.8117647,0.043137256],
    //                                         [0.5372549, 0.14117648, 0.25490198],
    //                                         [0.05490196, 0.15686275,1.0],
    //                                         [0.023529412,0.9019608,0.21568628],
    //                                         [0.07450981,0.5764706, 0.75686276],
    //                                         [0.19215687, 0.98039216,1.0],
    //                                         [0.73333335,0.1764706, 0.49019608],
    //                                         [1.0, 0.0, 0.33333334],
    //                                         [ 0.9647059, 0.41960785, 0.41960785]]
    //
    //    var trainingOutputs: Tensor<Int32> = [1, 1, 0, 2, 1, 2, 2, 0, 0, 0]
    
    var trainingInputs: Tensor<Float> = [[0.6862745, 0.93333334, 0.58431375],
                                         [ 0.34901962,0.8117647,0.043137256],
                                         [0.5372549, 0.14117648, 0.25490198]]
    
    var trainingOutputs: Tensor<Int32> = [1, 1, 0]
    
    var inputs = [Tensor<Float>]()
    var outputs = [Tensor<Int32>]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        rField.delegate = self
        gField.delegate = self
        bField.delegate = self
        
        colorViewer.wantsLayer = true
        
        //        for (i,item) in trainingInputs.enumerated() {
        //            //            let mapped = item.map { $0 * 255.0 }
        //            inputs.append(Tensor(item).reshaped(to: [3]))
        //            outputs.append(Tensor(trainingOutputs[i]))
        //        }
        //
        //        let x: Tensor<Float> = [[1,2,3],[4,5,6]]
        //
        //        for (i,val) in inputs.enumerated() {
        //            print(val,outputs[i],val.rank)
        //        }
    }
    
    func controlTextDidChange(_ obj: Notification) {
        let r = CGFloat(Float(rField.stringValue) ?? 0.0) / CGFloat(255)
        let g = CGFloat(Float(gField.stringValue) ?? 0.0) / CGFloat(255)
        let b = CGFloat(Float(bField.stringValue) ?? 0.0) / CGFloat(255)
        
        colorViewer.layer?.backgroundColor = CGColor(red: r, green: g, blue: b, alpha: 1)
    }
    
    @IBAction func generateColor(_ sender: Any) {
        
        let x = Int.random(in: 0...255)
        let y = Int.random(in: 0...255)
        let z = Int.random(in: 0...255)
        
        rField.stringValue = "\(x)"
        gField.stringValue = "\(y)"
        bField.stringValue = "\(z)"
        
        let r = CGFloat(Float(x) / 255.0)
        let g = CGFloat(Float(y) / 255.0)
        let b = CGFloat(Float(z) / 255.0)
        
        colorViewer.layer?.backgroundColor = CGColor(red: r, green: g, blue: b, alpha: 1)
    }
    
    @IBAction func save(_ sender: Any) {
        //        for case let element? in [rField, gField, bField] {
        //            if element.stringValue.isEmpty {
        //                return
        //            }
        //        }
        //
        //        let r = Float(rField.stringValue)! / 255.0
        //        let g = Float(gField.stringValue)! / 255.0
        //        let b = Float(bField.stringValue)! / 255.0
        //
        //        //        let tensor = Tensor([r,g,b])
        //        //        trainingInputs.append(tensor)
        //        //
        //        //        let output = Tensor(Int32(labelPicker.indexOfSelectedItem))
        //        //        trainingOutputs.append(output)
        //
        //        [rField, gField, bField].forEach { (element) in
        //            element?.stringValue = ""
        //        }
        //
        //        print(trainingInputs)
        //        print(trainingOutputs)
        //        print("")
    }
    
    func prepData() {
        
    }
    
    @IBAction func train(_ sender: Any) {
        
        let optimizer = SGD(for: model, learningRate: learningRate)
        let (loss, grad) = valueWithGradient(at: model) { [weak self] (model) -> Tensor<Float> in
            return softmaxCrossEntropy(logits: model(self!.trainingInputs), labels: self!.trainingOutputs)
        }
        
        optimizer.update(&model, along: grad)
        print(loss)
//
//        for _ in 1...epochs {
//
//            let (loss, grad) = valueWithGradient(at: model) { (model) -> Tensor<Float> in
//
//                return softmaxCrossEntropy(logits: model(self.trainingInputs), labels: self.trainingOutputs)
//            }
//
//            optimizer.update(&model, along: grad)
//            print(loss)
//        }
        
    }
    
    @IBAction func classify(_ sender: Any) {
        
        let x = Float(rField.stringValue)!
        let y = Float(gField.stringValue)!
        let z = Float(bField.stringValue)!
        
        rField.stringValue = "\(String(describing: x))"
        gField.stringValue = "\(String(describing: y))"
        bField.stringValue = "\(String(describing: z))"
        
        let r = CGFloat(x / 255.0)
        let g = CGFloat(y / 255.0)
        let b = CGFloat(z / 255.0)
        
        colorViewer.layer?.backgroundColor = CGColor(red: r, green: g, blue: b, alpha: 1)
        
        let val = Tensor([x/255, y/255,z/255]).reshaped(to: [1,3])
        let prediction = model.callAsFunction(val)
        
        let res = prediction.argmax(squeezingAxis: 1).scalarized()
        let label = ["Red", "Green", "Blue"][Int(res)]
        outputLabel.stringValue = "Color: " + label
    }
    
    func accuracy(predictions: Tensor<Int32>, truths: Tensor<Int32>) -> Float {
        return Tensor<Float>(predictions .== truths).mean().scalarized()
    }
    
}
