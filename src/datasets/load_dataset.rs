use std::path::PathBuf;

use hf_hub::api::sync::Api;
use std::fs::File;
use arrow::{
    record_batch::RecordBatch,
    datatypes::Schema,
};
use parquet::{
    arrow::{ArrowReader, ArrowWriter, ParquetFileArrowReader},
    file::reader::SerializedFileReader,
};
use std::sync::Arc;

pub enum DatasetFileType {
    PARQUET,
    CSV,
    JSON,
    TSV,
    TXT
}

pub struct Dataset {
    name_or_path: String,
    api: Api,
    train_file: String,
    test_file: String,
    filetype: DatasetFileType
}

pub struct DataTable{
    pub schema: Schema,
    pub data: Vec<RecordBatch>,
    pub rows: usize,
}


impl Dataset {
    pub fn new(name_or_path: &str, train_file: &str, test_file: &str, filetype: &str) -> Self {
        let api = Api::new().unwrap();
        Self {
            name_or_path: name_or_path.to_string(),
            api,
            train_file: train_file.to_string(),
            test_file: test_file.to_string(),
            filetype: match filetype {
                "parquet" => DatasetFileType::PARQUET,
                "csv" => DatasetFileType::CSV,
                "json" => DatasetFileType::JSON,
                "tsv" => DatasetFileType::TSV,
                "txt" => DatasetFileType::TXT,
                _ => DatasetFileType::PARQUET
            }
        }
    }

    pub fn download(&self) -> Result<Vec<PathBuf>, anyhow::Error> {
        let test_filename = self.api
            .dataset(self.name_or_path.clone())
            .get(&self.test_file)
            .unwrap();

        let train_filename = self.api
            .dataset(self.name_or_path.clone())
            .get(&self.train_file)
            .unwrap();

        let filenames: Vec<PathBuf> = vec![train_filename, test_filename];

        Ok(filenames)   
    }

    pub fn load_parquet(&self, filename: &PathBuf) -> Result<DataTable, anyhow::Error> {
        let file = File::open(filename)?;
        let file_reader = SerializedFileReader::new(file).unwrap();
        let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));

        let schema = arrow_reader.get_schema().unwrap();
        let record_batch_reader = arrow_reader.get_record_reader(1).unwrap();
        let mut data: Vec<RecordBatch> = Vec::new();

        let mut rows = 0;
        for maybe_batch in record_batch_reader {
            let record_batch = maybe_batch.unwrap();
            rows += record_batch.num_rows();

            data.push(record_batch);
        }

        
        Ok(DataTable {
            schema,
            data,
            rows
        })
    }
}