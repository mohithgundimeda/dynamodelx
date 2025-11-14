from .ufa import UFA

def main(instance: UFA):
    ufa = UFA(
        task='regression', 
        model_size=None, 
        input_dim=5, 
        output_dim=1, 
        device='cpu', 
        custom_architecture=[2], 
        weights_init='he', 
        hidden_activation='relu',
        optimizer='adam',
        
        )

if __name__ == "__main__":
    main(UFA)