//
//  DataCollectionView.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/1/24.
//

import SwiftUI

struct DataCollectionView: View {
    
    @Bindable var bluetoothTool: BluetoothTool
    var trainingData: TrainingData
    
    var body: some View {
        
        NavigationStack {
            VStack {
                Section {
                    CurrentRoom(bluetoothTool: bluetoothTool)
                }
                Spacer()
                Section {
                    CollectionButton(bluetoothTool: bluetoothTool)
                        .disabled(bluetoothTool.currentRoom.isEmpty)
                }
                Spacer()
            }
            .navigationTitle("Training")
            .navigationBarTitleDisplayMode(.automatic)
            .toolbar {
                NavigationLink(destination: DataView(trainingData: trainingData)) {
                    Text("Data")
                }
            }
            .alert("Data collected", isPresented: $bluetoothTool.trainingFinished) {
                Button("Submit") {
                    bluetoothTool.saveData(to: trainingData)
                    
                    // Clear data for new run
                    bluetoothTool.clearData()
                }
                Button("Cancel", role: .cancel) {
                    bluetoothTool.clearData()
                }
            } message: {
                Text("\(bluetoothTool.peripherals.count) Unique Data Points Captured for \(bluetoothTool.currentRoom)")
            }
        }
        .padding(.all, 20)
    }
}

#if DEBUG
#Preview {
    DataCollectionView(bluetoothTool: BluetoothTool(), trainingData: TrainingData.example)

}
#endif
