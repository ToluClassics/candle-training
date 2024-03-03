use tokenizers::tokenizer::Tokenizer;
use candle_core::{Tensor, Device};
use anyhow::Error as E;

pub fn tokenize_dataset(input_strings: Vec<String>, tokenizer: &Tokenizer, device: &Device ) -> Result<(Tensor, Tensor), anyhow::Error> {
    let tokens = tokenizer
            .encode_batch(input_strings, true)
            .map_err(E::msg)?;
    
    let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), device)?)
            })
            .collect::<Result<Vec<_>, E>>()?;
    
    let token_ids = Tensor::stack(&token_ids, 0)?;
    let token_type_ids = token_ids.zeros_like()?;

    Ok((token_ids, token_type_ids))

}

pub fn process_entire_dataset(input_strings: Vec<String>, tokenizer: Tokenizer, device: Device) -> Result<(Tensor, Tensor), anyhow::Error> {
    let mut entire_tensor: Vec<Tensor> = vec![];
    let mut entire_tensor_type_ids: Vec<Tensor> = vec![];

    // iterate through input_strings in batches

    for batch in input_strings.chunks(8) {
        let (token_ids, token_type_ids) = match tokenize_dataset(batch.to_vec(), &tokenizer, &device) {
            Ok(result) => result,
            Err(error) => return Err(error),
        };

        entire_tensor.push(token_ids);
        entire_tensor_type_ids.push(token_type_ids);
        
    }

    let entire_tensor = Tensor::cat(&entire_tensor, 0)?;
    let entire_tensor_type_ids = Tensor::cat(&entire_tensor_type_ids, 0)?;

    Ok((entire_tensor, entire_tensor_type_ids))
}