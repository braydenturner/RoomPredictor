//
//  ContentView.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 2/9/24.
//

import SwiftUI

struct ContentView: View {
    
    var bluetoothTool: BluetoothTool
    
    var body: some View {
        TabView {
            Tab("Predict", systemImage: "square.split.bottomrightquarter") {
                PredictView()
            }
            Tab("Train", systemImage: "rectangle.and.pencil.and.ellipsis") {
                DataCollectionView(bluetoothTool:bluetoothTool, trainingData: TrainingData())
            }
        }
    }
}

#Preview {
    ContentView(bluetoothTool: BluetoothTool())
}
