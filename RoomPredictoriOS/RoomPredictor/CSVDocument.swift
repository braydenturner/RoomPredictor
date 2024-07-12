//
//  CSVDocument.swift
//  RoomPredictor
//
//  Created by Brayden Turner on 7/3/24.
//

import SwiftUI
import UniformTypeIdentifiers

struct CSVDocument: FileDocument {
    static var readableContentTypes: [UTType] {
        [.commaSeparatedText]
    }
    
    var text = ""
    
    init(text: String) {
        self.text = text
    }
    
    init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            text = String(decoding: data, as: UTF8.self)
        } else {
            text = ""
        }
    }
    
    init(from data: TrainingData) {
        text.append(TrainingData.header.joined(separator: ",") + "\n")
        
        for room in data.rooms {
            for point in room.points{
                text.append("\(room.room),\(point.id),\(point.rssi),\(point.time),\(point.name)\n")
            }
        }
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}
