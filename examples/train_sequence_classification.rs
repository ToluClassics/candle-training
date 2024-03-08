use arrow::array::{Int64Array, StringArray};
use clap::Parser;
use log::info;
use env_logger;
use rand::rngs::ThreadRng;
use kdam::tqdm;
use anyhow::Result;

use candle_core::{DType, Device, Tensor};
use candle_nn::{ AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};

use candle_training::datasets::load_dataset::Dataset;
use candle_training::models::load_model::get_config_tokenizer_path;
use candle_training::models::roberta::{RobertaForSequenceClassification, RobertaConfig};
use rand::seq::SliceRandom; // Import the SliceRandom trait

/*
RUST_LOG=info  cargo run --release --example train_sequence_classification --  --dataset-name imdb --hf-train-file plain_text/train-00000-of-00001.parquet --hf-test-file plain_text/test-00000-of-00001.parquet --max-length 128 --pad-to-max-length --model-name wrmurray/roberta-base-finetuned-imdb  --train-batch-size 8  --eval-batch-size 8 --learning-rate 5e-5 --weight-decay 0.0 --num-train-epochs 3 --gradient-accumulation-steps 1
 */

pub const FLOATING_DTYPE: DType = DType::F32;
pub const LONG_DTYPE: DType = DType::I64;
const EPOCHS: usize = 10;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args{
    /// name of the dataset on hf_hub
    #[arg(short, long)]
    dataset_name: String,

    /// name or path to the train file on hf_hub
    #[arg(long)]
    hf_train_file: String,

    /// name or path to the test file on hf_hub
    #[arg(long)]
    hf_test_file: String,

    /// max length of the input
    #[arg(short, long, default_value_t = 128)]
    max_length: u16,

    /// pad to max length
    #[arg(short, long, default_value_t = true)]
    pad_to_max_length: bool,

    /// name of the model on hf_hub
    #[arg(long)]
    model_name: String,

    /// Training batch size
    #[arg(short, long)]
    train_batch_size: usize,

    /// Evaluation batch size
    #[arg(short, long)]
    eval_batch_size: usize,

    /// learning rate
    #[arg(short, long, default_value_t = 5e-5)]
    learning_rate: f64,

    /// Weight decay
    #[arg(short, long, default_value_t = 0.0)]
    weight_decay: f32,

    /// Number of epochs
    #[arg(short, long, default_value_t = 3)]
    num_train_epochs: u16,

    /// Maximum train steps
    #[arg(long)]
    max_train_steps: Option<u16>,

    /// Gradient accumulation steps
    #[arg(short, long, default_value_t = 1)]
    gradient_accumulation_steps: u16,




}
pub fn main() -> Result<()>{

    // Initialize logger
    env_logger::init();

    let args = Args::parse();
    let mut seed: ThreadRng = rand::thread_rng();
    let device = Device::cuda_if_available(0)?;

    // Set up logging
    info!("Logging training parameters");
    info!("Dataset Name: {}", args.dataset_name);
    info!("HF Train File: {}", args.hf_train_file);
    info!("HF Test File: {}", args.hf_test_file);
    info!("Max Length: {}", args.max_length);
    info!("Pad to Max Length: {}", args.pad_to_max_length);
    info!("Model Name: {}", args.model_name);
    info!("Train Batch Size: {}", args.train_batch_size);
    info!("Eval Batch Size: {}", args.eval_batch_size);
    info!("Learning Rate: {}", args.learning_rate);
    info!("Weight Decay: {}", args.weight_decay);
    info!("Num Train Epochs: {}", args.num_train_epochs);
    info!{"Gradient Accumulation Steps: {}", args.gradient_accumulation_steps};

    let dataset = Dataset::new(&args.dataset_name, 
        &args.hf_train_file,
        &args.hf_test_file, 
        "parquet");

    let training_file_paths = dataset.download();
    let training_file_paths = training_file_paths?;

    let train_datatable = dataset.load_parquet(&training_file_paths[0], args.train_batch_size)?;
    let test_datatable = dataset.load_parquet(&training_file_paths[1], args.eval_batch_size)?;

    let mut train_strings: Vec<String> = Vec::new();
    let mut train_labels: Vec<i64> = Vec::new();

    let mut test_strings: Vec<String> = Vec::new();
    let mut test_labels: Vec<i64> = Vec::new();

    for record_batch in train_datatable.data{
        let string_array = record_batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let label_array = record_batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

        
        for i in 0..string_array.iter().len(){
            train_strings.push(string_array.value(i).to_string());
            train_labels.push(label_array.value(i));
        }
    }

    for record_batch in test_datatable.data{
        let string_array = record_batch.column(0).as_any().downcast_ref::<StringArray>().unwrap();
        let label_array = record_batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

        
        for i in 0..string_array.iter().len(){
            test_strings.push(string_array.value(i).to_string());
            test_labels.push(label_array.value(i));
        }
    }

    info!("Number of training examples: {}", train_strings.len());
    info!("Number of test examples: {}", test_strings.len());
    info!("Sample Training example: {:?} and Label: {:?}", train_strings[0], train_labels[0]);

    let mut indices = (0..train_strings.len()).collect::<Vec<usize>>();
    indices.shuffle(&mut seed); // Use the shuffle method from the SliceRandom trait

    let train_strings: Vec<String> = indices.iter().map(|i| train_strings[*i].clone()).collect();
    let train_labels: Vec<i64> = indices.iter().map(|i| train_labels[*i]).collect();

    let mut test_indices = (0..test_strings.len()).collect::<Vec<usize>>();
    test_indices.shuffle(&mut seed); // Use the shuffle method from the SliceRandom trait

    // select a subset of the test data about 1000 examples
    test_indices = test_indices.iter().take(1000).map(|i| *i).collect();

    let test_strings: Vec<String> = test_indices.iter().map(|i| test_strings[*i].clone()).collect();
    let test_labels: Vec<i64> = test_indices.iter().map(|i| test_labels[*i]).collect();

    let (config_file, tokenizer, weights_filename) = get_config_tokenizer_path(&args.model_name, false)?;
    let mut config: RobertaConfig = serde_json::from_str(&config_file)?;
    config._num_labels = Some(2);
    config.problem_type = Some("single_label_classification".to_string());

    let mut varmap = VarMap::new();
    // let mut vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let vs = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename.clone()], FLOATING_DTYPE, &device)? };
    varmap.load(weights_filename)?;
    
    let model = RobertaForSequenceClassification::load(vs, &config)?;
    let mut paramsw = ParamsAdamW::default();
    paramsw.lr = args.learning_rate;
    let mut adamw = AdamW::new(varmap.all_vars(), paramsw)?;

    for epoch in 1..EPOCHS + 1 {
        info!("Epoch: {}", epoch);

        for (_i, (input, label)) in tqdm!(train_strings.chunks(args.train_batch_size).zip(train_labels.chunks(args.train_batch_size)).enumerate()){

            let pad_token_id = config.pad_token_id as u32;
            let max_len = 128;

            let tokens = tokenizer.encode_batch(input.to_vec(), true).unwrap();
            let token_ids = tokens
            .iter()
            .map(|tokens| {
                let mut tokens = tokens.get_ids().to_vec();
                tokens.resize(max_len, pad_token_id);
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;

            let token_ids = Tensor::stack(&token_ids, 0)?;
            let token_type_ids = token_ids.zeros_like()?;
            let labels = Tensor::new(label, &device)?;

            let outputs = model.forward(&token_ids, &token_type_ids, Some(&labels))?;
            let loss = outputs.loss.unwrap();

            adamw.backward_step(&loss)?;

            if _i % 100 == 0 {
                info!("Loss: {:?}", loss.to_device(&Device::Cpu)?.to_scalar::<f32>()?);
            };

            //compute test accuracy
            if _i % 500 == 0 {
                let mut correct = 0;
                let mut total = 0;
                for (_j, (test_input, test_label)) in test_strings.chunks(args.eval_batch_size).zip(test_labels.chunks(args.eval_batch_size)).enumerate(){
                    let tokens = tokenizer.encode_batch(test_input.to_vec(), true).unwrap();
                    let token_ids = tokens
                        .iter()
                        .map(|tokens| {
                            let mut tokens = tokens.get_ids().to_vec();
                            tokens.resize(max_len, pad_token_id);
                            Ok(Tensor::new(tokens.as_slice(), &device)?)
                        })
                        .collect::<Result<Vec<_>>>()?;

                    let token_ids = Tensor::stack(&token_ids, 0)?;
                    let token_type_ids = token_ids.zeros_like()?;
                    let test_label = Tensor::new(test_label, &device)?;

                    let outputs = model.forward(&token_ids, &token_type_ids, Some(&test_label))?;
                    let logits = outputs.logits.to_device(&Device::Cpu)?.to_vec2::<f32>().unwrap();
                    let labels = test_label.to_vec1::<i64>().unwrap();

                    //for vec in logits get the maximum index
                    let mut k=0;
                    for vec in logits{
                        let max_index = vec.iter().enumerate().max_by(|(_, u), (_, v)| u.total_cmp(v))
                        .map(|(i, _)| i as i64)
                        .unwrap();

                        if max_index == labels[k]{
                            correct += 1;
                        }
                        total += 1;
                        k+=1;
                    }
                }
                info!("Test Accuracy: {}", correct as f32 / total as f32);
            }
        
        }

        }

        Ok(())
    }