//
//  PredictView.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/1/24.
//

import SwiftUI
import CoreML

struct PredictView: View {
    
    let classifier = try! RoomClassifier()
    
    var body: some View {
        NavigationStack {
            VStack {
                
            }
            .navigationTitle("Predict")
        }
        .padding(.all, 20)
    }
    
}

#if DEBUG
#Preview {
    PredictView()
}
#endif
