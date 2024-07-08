//
//  DataView.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 7/3/24.
//

import SwiftUI

struct DataView: View {
    
    @EnvironmentObject var trainingData: TrainingData
    
    @State var submit: Bool = false
    @State var exporting: Bool = false
    
    var body: some View {
        VStack {
            List {
                ForEach(trainingData.rooms) { room in
                    HStack {
                        Text(room.room)
                        Spacer()
                        Text("\(room.points.count)")
                    }
                }.onDelete { indexSet in
                    trainingData.rooms.remove(atOffsets: indexSet)
                }
            }
        }
        .navigationTitle("Count")
        .toolbar {
            Button("Export") {
                self.exporting.toggle()
            }.fileExporter(
                isPresented: $exporting,
                document: CSVDocument(from: trainingData),
                contentType: .commaSeparatedText,
                defaultFilename: "roomPredictionData.csv"
            ) { result in
                switch result {
                case .success(let file):
                    print(file)
                case .failure(let error):
                    print(error)
                }
                
                trainingData.clearRooms()
            }
        }
    }
}

#if DEBUG
#Preview {
    DataView()
        .environmentObject(TrainingData.example)
}
#endif
