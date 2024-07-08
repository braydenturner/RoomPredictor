//
//  CollectionButton.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 3/2/24.
//

import SwiftUI
import UIKit

struct CollectionButton: View {
    
    @EnvironmentObject var bluetoothTool: BluetoothTool
    
    @State private var animationAmount = 1.0
    
    enum ButtonText: String {
        case start = "Start"
        case stop = "Stop"
    }
    
    var body: some View {
        VStack {
            if bluetoothTool.collecting {
                // Stop Collecting
                Button(ButtonText.stop.rawValue) {
                    bluetoothTool.trainingFinished.toggle()
                    bluetoothTool.collecting.toggle()
                    bluetoothTool.stopScanning()
                }
                .buttonStyle(CollectionButtonStyle())
                .overlay(
                    Circle()
                        .stroke(.blue)
                        .scaleEffect(animationAmount)
                        .opacity(2 - animationAmount)
                        .animation(
                            .easeInOut(duration: 1)
                            .repeatForever(autoreverses: false),
                            value: animationAmount
                        )
                )
                .onAppear {
                    animationAmount = 2
                }
            } else {
                // Start collecting
                Button(ButtonText.start.rawValue) {
                    bluetoothTool.cleanRoomName()
                    bluetoothTool.collecting.toggle()
                    bluetoothTool.scanForPeripherals()
                    hideKeyboard()
                }
                .buttonStyle(CollectionButtonStyle())
            }
        }
    }
}

struct CollectionButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(50)
            .background(.blue)
            .foregroundStyle(.white)
            .clipShape(.circle)
        
    }
}

#if canImport(UIKit)
extension View {
    func hideKeyboard() {
        UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
    }
}
#endif

#Preview {
    CollectionButton()
        .environmentObject(BluetoothTool())
}
