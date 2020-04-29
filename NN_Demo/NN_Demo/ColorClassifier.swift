//
//  ColorClassifier.swift
//  NN_Demo
//
//  Created by Cyril Garcia on 4/28/20.
//  Copyright © 2020 Cyril Garcia. All rights reserved.
//

import Cocoa
import TensorFlow
#if canImport(PythonKit)
import PythonKit
#else
import Python
#endif

final class ColorClassifier: NSViewController {
    
    @IBOutlet var redSlider: NSSlider!
    @IBOutlet var greenSlider: NSSlider!
    @IBOutlet var blueSlider: NSSlider!
    @IBOutlet var labelSelector: NSPopUpButton!
    @IBOutlet var outputLabel: NSTextField!
    @IBOutlet var colorView: NSView!
    
    let classificationLabels = ["Red","Blue","Green","Yellow","Purple","Brown","Pink","black","White"]
    
    var weights = Tensor(shape: [3,9],scalars: [Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1),
                                                Float.random(in: 0...1),Float.random(in: 0...1),Float.random(in: 0...1)])
        
    var raw_inputs = [Float]()
    var raw_outputs = [Float]()
    
    var trainingInput: Tensor<Float>!
    var trainingOutput: Tensor<Float>!
    
//    var testInput = Tensor(shape: [], scalars: <#T##[_]#>)
    
    override func viewDidLoad() {
        super.viewDidLoad()
//        let r = CGFloat.random(in: 0...1)
//        let g = CGFloat.random(in: 0...1)
//        let b = CGFloat.random(in: 0...1)

        labelSelector.removeAllItems()
        labelSelector.addItems(withTitles: classificationLabels)
    }
    
    @IBAction func train(_ sender: Any) {
        raw_inputs = [0.58457386, 0.873974, 0.2523505, 0.7349865, 0.11772124, 0.1399177, 0.091900155, 0.96005094, 0.30265996, 0.43436098, 0.2061948, 0.84002817, 0.6674902, 0.2572348, 0.4988692, 0.8522466, 0.044111863, 0.11403932, 0.4911531, 0.2768156, 0.7581087, 0.5256973, 0.9096914, 0.64478236, 0.013603632, 0.47027507, 0.71730167, 0.14791192, 0.27054757, 1.0, 0.14791192, 0.27054757, 0.43677083, 0.14791192, 0.90336007, 0.43677083, 0.14791192, 0.90336007, 0.08145834, 1.0, 0.90336007, 0.08145834, 1.0, 0.90336007, 0.41609374, 1.0, 1.0, 0.61364585, 1.0, 1.0, 0.0, 1.0, 1.0, 0.19864583, 1.0, 0.8985937, 0.19864583, 0.50216556, 0.020547176, 0.4865946, 0.50216556, 0.020547176, 0.0, 1.0, 0.020547176, 0.0, 0.14614584, 0.020547176, 0.0, 0.45239583, 0.020547176, 0.0, 0.90593755, 0.2203125, 0.0, 0.0, 0.0, 0.0, 0.0, 0.034635417, 0.0575, 1.0, 1.0, 1.0, 1.0, 0.93609375, 1.0, 0.94463545, 0.93609375, 1.0, 0.94463545, 0.93609375, 0.9115104, 0.94463545, 0.8410417, 0.9115104, 0.575625, 0.43520835, 0.04921875, 0.5276844, 0.35780025, 0.21673357, 0.5276844, 0.40316483, 0.21673357, 0.5911219, 0.40316483, 0.21673357, 0.5911219, 0.40316483, 0.29829606, 0.5911219, 0.40316483, 0.15142109, 0.5911219, 0.40316483, 0.0, 0.7576323, 0.37868565, 0.0, 0.5296115, 0.37868565, 0.0]
        
        raw_outputs = [2, 0, 2, 4, 6, 0, 4, 2, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 3, 4, 0, 0, 0, 0, 0, 7, 7, 8, 8, 8, 8, 8, 3, 5, 5, 5, 5, 5, 5, 5, 5]
        
        trainingInput = Tensor(shape: [raw_inputs.count / 3,3], scalars: raw_inputs)
        trainingOutput = Tensor(shape: [raw_outputs.count, 1], scalars: raw_outputs)
        print(trainingInput)
        print(weights)
        
        let yhat = softmax(trainingInput • weights)
        
        print(softmaxCrossEntropy(logits: yhat, probabilities: trainingOutput))
        
    }
    
    func predict(_ x: Tensor<Float>) {
        
      
        //        let sgd = SGD(for: <#T##_#>, learningRate: <#T##Float#>, momentum: <#T##Float#>, decay: <#T##Float#>, nesterov: <#T##Bool#>)
            
            
            //        let val = softmax(x • weights).sum()
            
        }
    
    @IBAction func saveInput(_ sender: Any) {
        let r = redSlider.floatValue / 255.0
        let g = greenSlider.floatValue / 255.0
        let b = blueSlider.floatValue / 255.0
        let output = labelSelector.indexOfSelectedItem
        
        raw_inputs.append(contentsOf: [r,g,b])
        raw_outputs.append(Float(output))
    }
    
    @IBAction func randomColor(_ sender: Any) {
        let r = CGFloat.random(in: 0...1)
        let g = CGFloat.random(in: 0...1)
        let b = CGFloat.random(in: 0...1)
        setColor(r,g,b)
        redSlider.floatValue = Float(r * 255.0)
        greenSlider.floatValue = Float(g * 255.0)
        blueSlider.floatValue = Float(b * 255.0)
        
    }
    
    @IBAction func sliderDidChange(_ sender: NSSlider) {
        
        let r = CGFloat(redSlider.floatValue / 255.0)
        let g = CGFloat(greenSlider.floatValue / 255.0)
        let b = CGFloat(blueSlider.floatValue / 255.0)
        setColor(r,g,b)
    }
    
    func setColor(_ r: CGFloat,_ g: CGFloat,_ b: CGFloat) {
        colorView.wantsLayer = true
        colorView.layer?.backgroundColor = CGColor(red: r, green: g, blue: b, alpha: 1)
    }
}
