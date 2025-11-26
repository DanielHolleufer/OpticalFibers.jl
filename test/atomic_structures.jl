using Test
using OpticalFibers

@testset "NLevelAtom isequal and hash" begin
    γ_1 = 1.0
    d_1 = [1.0, 0.0, 0.0]
    two_level_atom_1 = TwoLevelAtom(γ_1, d_1)

    γ_2 = 1.0
    d_2 = [1.0, 0.0, 0.0]
    two_level_atom_2 = TwoLevelAtom(γ_2, d_2)

    γ_3 = 2.0
    d_3 = [1.0, 0.0, 0.0]
    two_level_atom_3 = TwoLevelAtom(γ_3, d_3)

    γ_4 = 1.0
    d_4 = [0.0, 1.0, 0.0]
    two_level_atom_4 = TwoLevelAtom(γ_4, d_4)
    
    @test isequal(two_level_atom_1, two_level_atom_2)
    @test isequal(hash(two_level_atom_1), hash(two_level_atom_2))

    @test !isequal(two_level_atom_1, two_level_atom_3)
    @test !isequal(two_level_atom_1, two_level_atom_4)
    @test !isequal(hash(two_level_atom_1), hash(two_level_atom_3))
    @test !isequal(hash(two_level_atom_1), hash(two_level_atom_4))

    Δ_1 = 0.0
    γ_lower_1 = 1.0
    γ_upper_1 = 0.1
    d_lower_1 = [1.0, 0.0, 0.0]
    d_upper_1 = [0.0, 1.0, 0.0]
    three_level_atom_1 = ThreeLevelAtom(γ_lower_1, d_lower_1, γ_upper_1, d_upper_1, Δ_1)
    three_level_atom_1_lower = TwoLevelAtom(γ_lower_1, d_lower_1)
    three_level_atom_1_upper = TwoLevelAtom(γ_upper_1, d_upper_1)
    three_level_atom_1a = ThreeLevelAtom(
        three_level_atom_1_lower, three_level_atom_1_upper, Δ_1
    )

    Δ_2 = 1.0
    γ_lower_2 = 1.0
    γ_upper_2 = 0.1
    d_lower_2 = [1.0, 0.0, 0.0]
    d_upper_2 = [0.0, 1.0, 0.0]
    three_level_atom_2 = ThreeLevelAtom(γ_lower_2, d_lower_2, γ_upper_2, d_upper_2, Δ_2)

    Δ_3 = 0.0
    γ_lower_3 = 2.0
    γ_upper_3 = 0.1
    d_lower_3 = [1.0, 0.0, 0.0]
    d_upper_3 = [0.0, 1.0, 0.0]
    three_level_atom_3 = ThreeLevelAtom(γ_lower_3, d_lower_3, γ_upper_3, d_upper_3, Δ_3)

    Δ_4 = 0.0
    γ_lower_4 = 1.0
    γ_upper_4 = 0.5
    d_lower_4 = [1.0, 0.0, 0.0]
    d_upper_4 = [0.0, 1.0, 0.0]
    three_level_atom_4 = ThreeLevelAtom(γ_lower_4, d_lower_4, γ_upper_4, d_upper_4, Δ_4)

    Δ_5 = 0.0
    γ_lower_5 = 1.0
    γ_upper_5 = 0.1
    d_lower_5 = [0.0, 1.0, 0.0]
    d_upper_5 = [0.0, 1.0, 0.0]
    three_level_atom_5 = ThreeLevelAtom(γ_lower_5, d_lower_5, γ_upper_5, d_upper_5, Δ_5)

    Δ_6 = 0.0
    γ_lower_6 = 1.0
    γ_upper_6 = 0.1
    d_lower_6 = [1.0, 0.0, 0.0]
    d_upper_6 = [0.0, -1.0, 0.0]
    three_level_atom_6 = ThreeLevelAtom(γ_lower_6, d_lower_6, γ_upper_6, d_upper_6, Δ_6)

    @test isequal(three_level_atom_1, three_level_atom_1)
    @test isequal(hash(three_level_atom_1), hash(three_level_atom_1))
    @test isequal(three_level_atom_1, three_level_atom_1a)

    @test !isequal(three_level_atom_1, three_level_atom_2)
    @test !isequal(three_level_atom_1, three_level_atom_3)
    @test !isequal(three_level_atom_1, three_level_atom_4)
    @test !isequal(three_level_atom_1, three_level_atom_5)
    @test !isequal(three_level_atom_1, three_level_atom_6)
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_2))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_3))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_4))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_5))
    @test !isequal(hash(three_level_atom_1), hash(three_level_atom_6))
end
