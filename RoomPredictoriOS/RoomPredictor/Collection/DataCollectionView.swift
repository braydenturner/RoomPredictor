//
//  DataCollectionView.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/1/24.
//

import SwiftUI

struct DataCollectionView: View {
    
    @EnvironmentObject var bluetoothTool: BluetoothTool
    @EnvironmentObject var trainingData: TrainingData
    
    var body: some View {
        NavigationStack {
            VStack {
                Section {
                    CurrentRoom()
                }
                Spacer()
                Section {
                    CollectionButton()
                        .disabled(bluetoothTool.currentRoom.isEmpty)
                }
                Spacer()
            }
            .navigationTitle("Training")
            .navigationBarTitleDisplayMode(.automatic)
            .toolbar {
                NavigationLink(destination: DataView()) {
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
    DataCollectionView()
        .environmentObject(BluetoothTool())
        .environmentObject(TrainingData.example)
}
#endif
