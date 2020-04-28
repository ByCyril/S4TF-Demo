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
    
    var weights = Tensor(shape: [9,1], scalars: [Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1),
                                                 Float.random(in: 0...1)])
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
 
        
    }
}
