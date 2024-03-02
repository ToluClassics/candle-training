#[cfg(test)]
mod tests {
    use candle_training::datasets::load_dataset::Dataset;

    #[test]
    fn test_load_imdb_dataset() {
        let dataset_name = "imdb";
        let train_file = "plain_text/train-00000-of-00001.parquet";
        let test_file = "plain_text/test-00000-of-00001.parquet";

        let dataset = Dataset::new(dataset_name, train_file, test_file, "parquet");
        let filename = dataset.download();
        let filenames = filename.unwrap();

        let train_datatable = dataset.load_parquet(&filenames[0]);
        assert_eq!(train_datatable.unwrap().rows, 25000);

        let test_datatable = dataset.load_parquet(&filenames[1]);
        assert_eq!(test_datatable.unwrap().rows, 25000);


    }

}